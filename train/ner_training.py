"""
BERT-BiLSTM-CRF 命名实体识别模型训练脚本
"""

import os
import json
import logging
import argparse
from types import SimpleNamespace
from dataclasses import dataclass, asdict, field
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
import sys
import shutil
import time
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模型
from models.NER_model import BertBiLSTMCRF, LABEL_MAP, NUM_LABELS

# 设置日志
def setup_logging(args):
    """设置日志记录"""
    # 创建日志目录
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建一个时间戳标记的日志文件名
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 配置日志处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 记录脚本启动信息
    logger.info("="*50)
    logger.info("MSRA NER训练脚本启动")
    logger.info("命令行参数: %s", vars(args))
    logger.info("="*50)
    
    return logger

logger = logging.getLogger(__name__)

# ================== 内置配置（无需命令行即可运行） ==================
# 如果只想直接运行:  python ner_training.py  使用下面的默认配置即可。
# 若仍希望用命令行覆盖，切换 USE_CLI = True；
# 当 USE_CLI = False 且命令行未提供其它参数时，会直接采用内置配置。

USE_CLI = False

@dataclass
class TrainConfig:
    # 数据参数
    train_file: str = r"/workspace/MRSA/data/splits/labeled_1%.json"
    eval_file: str | None = None  # 若为 None 自动切分
    output_dir: str = "../models/saved_models"
    # 模型参数
    bert_model: str = r"/workspace/MRSA/models/google-bert/bert-base-chinese"
    max_seq_length: int = 128
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1
    # 训练策略 (BERT 冻结策略简化为单参数)
    # freeze_bert_layers 含义:
    #   None 或 0 -> 不冻结
    #   正整数 N  -> 冻结前 N 层 encoder.layer (以及 embeddings)
    #   -1        -> 冻结全部 BERT 参数
    freeze_bert_layers: int | None = None
    batch_size: int = 32
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 20
    warmup_steps: int = 0
    logging_steps: int = 100
    save_steps: int = 1
    seed: int = 42
    # 早停
    patience: int = 5
    min_delta: float = 0.0001
    early_stopping: bool = True
    # 分组学习率/权重衰减（更显式）
    bert_lr: float = 2e-5
    other_lr: float = 1e-3
    bert_weight_decay: float = 0.1
    other_weight_decay: float = 0.1
    # 评估 / 其他可扩展参数留空位
    # Checkpoint 相关
    max_checkpoints_to_keep: int = 3  # 轮换保留的最大 checkpoint 数
    free_space_warn_gb: float = 0.5   # 剩余空间低于该阈值(GB)发出警告
    legacy_torch_save_fallback: bool = True  # 若新格式写失败, 回退旧格式
    max_best_models_to_keep: int = 2  # 仅保留最新(按F1最高)的若干个 best_model_f1_*
    delete_redundant_on_save: bool = True  # 保存后立即触发清理

def build_namespace_from_dataclass(dc: TrainConfig) -> SimpleNamespace:
    return SimpleNamespace(**asdict(dc))


class MSRANERDataset(Dataset):
    """MSRA NER数据集加载器"""

    def __init__(self, tokenizer, max_length=128, data_file=None, data=None):
        """
        初始化数据集
        
        Args:
            tokenizer: BERT tokenizer
            max_length: 序列最大长度
            data_file: 数据文件路径 (可选)
            data: 直接提供的数据列表 (可选)
        """
        self.tokenizer = tokenizer
        # 需要 Fast tokenizer 才支持 word_ids()，否则后续对齐会报错
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "当前 tokenizer 不是 Fast 版本，不支持 word_ids()。请确保: "
                "1) 安装 tokenizers 库: pip install -U tokenizers transformers\n"
                "2) 使用 AutoTokenizer.from_pretrained(..., use_fast=True)\n"
                "3) 若使用本地模型目录，确认其中包含 fast tokenizer 所需的 tokenizer.json 文件。"
            )
        self.max_length = max_length
        
        # 注意: 允许传入空列表作为数据
        if data is not None:
            self.data = self.process_raw_data(data)
        elif data_file:
            self.data = self.load_data(data_file)
        else:
            raise ValueError("必须提供 data_file 或 data")

    def load_data(self, data_file):
        """从文件加载训练数据并预处理"""
        logger.info(f"从 {data_file} 加载数据...")
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return self.process_raw_data(raw_data)

    def process_raw_data(self, raw_data):
        """处理原始数据"""
        processed_data = []
        for item in raw_data:
            # 直接使用数据中提供的tokens和labels
            if 'tokens' in item and 'labels' in item:
                text = item['text']
                tokens = item['tokens']
                labels = item['labels']
                
                processed_data.append({
                    'text': text,
                    'tokens': tokens,
                    'labels': labels
                })                
        
        logger.info(f"加载并处理了 {len(processed_data)} 个样本")
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        labels = item['labels']
        
        # 将tokens转换为BERT的input_ids
        # 注意：这里我们直接使用原始tokens，不再调用tokenizer对text进行分词
        # 而是使用tokenizer将tokens转换为input_ids
        
        # BERT分词
        encodings = self.tokenizer(
            tokens,
            is_split_into_words=True,  # 告诉tokenizer输入已经分词
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # 处理标签
        labels_tensor = torch.ones(self.max_length, dtype=torch.long) * -100  # 填充标签为-100
        
        if labels:
            # 转换标签为ID
            label_ids = [LABEL_MAP.get(label, LABEL_MAP['O']) for label in labels]
            
            # 获取每个原始token对应的分词后token的映射
            # word_ids() 返回每个子词对应的原始词索引
            word_ids = encodings.word_ids()
            
            # 设置标签，注意特殊tokens如[CLS], [SEP]的word_id为None
            previous_word_id = None
            for i, word_id in enumerate(word_ids):
                if word_id is None:  # 特殊token
                    labels_tensor[i] = -100
                elif word_id != previous_word_id:  # 新token的开始
                    labels_tensor[i] = label_ids[word_id]
                else:  # 同一token的子词部分
                    # 如果是B-XXX，子词应变为I-XXX
                    if label_ids[word_id] % 2 == 1:  # B标签是奇数ID
                        labels_tensor[i] = label_ids[word_id] + 1  # B转为I
                    else:
                        labels_tensor[i] = label_ids[word_id]  # 保持O或I不变
                previous_word_id = word_id
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }


def train(args):
    """训练模型"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载 Fast tokenizer，启用 use_fast 保证可调用 word_ids
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"加载的 tokenizer 不是 Fast 版本: {args.bert_model} 。请检查模型目录是否完整或升级 transformers。"
        )
    
    # 加载数据集，支持自动划分验证集
    if args.eval_file is None:
        logger.info("未指定eval_file，将自动从训练集划分10%作为验证集...")
        with open(args.train_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        from sklearn.model_selection import train_test_split
        train_data, eval_data = train_test_split(
            all_data, test_size=0.1, random_state=args.seed
        )
        # 直接使用内存中的数据创建Dataset，避免磁盘I/O
        train_dataset = MSRANERDataset(tokenizer=tokenizer, max_length=args.max_seq_length, data=train_data)
        eval_dataset = MSRANERDataset(tokenizer=tokenizer, max_length=args.max_seq_length, data=eval_data)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    else:
        train_dataset = MSRANERDataset(data_file=args.train_file, tokenizer=tokenizer, max_length=args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataset = MSRANERDataset(data_file=args.eval_file, tokenizer=tokenizer, max_length=args.max_seq_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    
    # 加载模型
    model = BertBiLSTMCRF(
        num_labels=NUM_LABELS,
        bert_model_name=args.bert_model,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        dropout_rate=args.dropout,
        max_length=args.max_seq_length
    )
    model.to(device)
    
    # 打印模型详细信息
    model_info = model.get_model_info()
    logger.info("模型信息:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")
    
    # ========== 单参数冻结策略 ========== 
    layers_setting = args.freeze_bert_layers
    num_total_layers = len(model.bert.encoder.layer)
    if layers_setting in (None, 0):
        logger.info("不冻结任何BERT层 (全部可训练)")
    elif layers_setting == -1:
        logger.info("冻结所有BERT层 (embeddings + 所有 encoder 层)")
        model.freeze_bert_layers()  # 传 None -> 全部
    elif layers_setting > 0:
        if layers_setting > num_total_layers:
            logger.warning(f"请求冻结 {layers_setting} 层, 超过总层数 {num_total_layers}, 自动截断为 {num_total_layers}")
            layers_setting = num_total_layers
        logger.info(f"冻结 BERT 前 {layers_setting} 层 (包括 embeddings )")
        model.freeze_bert_layers(layers_setting)
    else:
        logger.warning(f"收到未识别的 freeze_bert_layers={layers_setting} 数值, 视为不冻结")

    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"参数统计: total={total_params} trainable={trainable_params} frozen={total_params-trainable_params}")
    
    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    
    # BERT参数
    bert_params = [p for n, p in model.bert.named_parameters() if p.requires_grad]
    bert_params_decay = [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
    bert_params_no_decay = [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad]

    # 其他参数 (LSTM, Classifier, CRF)
    other_params = [p for n, p in model.named_parameters() if not n.startswith('bert') and p.requires_grad]
    other_params_decay = [p for n, p in model.named_parameters() if not n.startswith('bert') and not any(nd in n for nd in no_decay) and p.requires_grad]
    other_params_no_decay = [p for n, p in model.named_parameters() if not n.startswith('bert') and any(nd in n for nd in no_decay) and p.requires_grad]

    optimizer_grouped_parameters = [
        {'params': bert_params_decay,    'lr': args.bert_lr,  'weight_decay': args.bert_weight_decay},
        {'params': bert_params_no_decay, 'lr': args.bert_lr,  'weight_decay': 0.0},
        {'params': other_params_decay,   'lr': args.other_lr, 'weight_decay': args.other_weight_decay},
        {'params': other_params_no_decay,'lr': args.other_lr, 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    
    # 学习率调度器
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    # =================== 训练循环 ===================
    logger.info("***** 开始训练 *****")
    global_step = 0
    best_f1 = 0.0
    no_improvement_count = 0
    early_stop = False

    for epoch in trange(args.num_train_epochs, desc="Epoch"):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            if step % args.logging_steps == 0:
                logger.info(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(1, len(train_dataloader))
        logger.info(f"Epoch {epoch} 平均损失: {avg_epoch_loss:.4f}")

        # ---------- 验证评估 ----------
        if eval_dataloader:
            eval_metrics = evaluate(model, eval_dataloader, device)
            logger.info(f"评估结果: {eval_metrics}")
            current_f1 = eval_metrics['f1']

            if current_f1 > best_f1 + args.min_delta:
                logger.info(f"F1 提升 {best_f1:.4f} -> {current_f1:.4f}，保存最佳模型")
                best_f1 = current_f1
                save_model(model, args.output_dir, f"best_model_f1_{best_f1:.4f}")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                logger.info(f"F1 未提升，连续 {no_improvement_count} 轮无改进")

            if args.early_stopping and no_improvement_count >= args.patience:
                logger.info(f"早停触发：连续 {args.patience} 轮无提升，提前结束训练")
                early_stop = True
                break

        # ---------- 保存 checkpoint ----------
        if (epoch + 1) % args.save_steps == 0:
            save_model(model, args.output_dir, f"checkpoint-{epoch+1}")

        if early_stop:
            break

    # 训练结束，保存最终模型
    save_model(model, args.output_dir, "final_model")
    training_stats = {
        "best_f1": best_f1,
        "completed_epochs": epoch + 1,
        "early_stopped": early_stop,
        "global_steps": global_step
    }
    logger.info(f"训练统计信息: {training_stats}")
    logger.info("训练完成！")
    return training_stats


def evaluate(model, eval_dataloader, device):
    """评估模型，返回规范化后的字典，并提供更友好的日志展示"""
    model.eval()

    true_labels_ids: list[list[int]] = []
    pred_labels_ids: list[list[int]] = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']  # shape: (B, L)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            predictions = model.decode(logits, attention_mask)  # list[list[int]] length L each

        # 对每个样本按有效标签位置对齐，避免 CLS/SEP / PAD 导致的位移
        for i in range(attention_mask.size(0)):
            label_row = labels[i]
            valid_pos = torch.nonzero(label_row != -100, as_tuple=False).flatten().tolist()
            if not valid_pos:
                continue  # 全部是 -100，跳过
            true_seq = label_row[valid_pos].cpu().numpy()
            pred_seq = [predictions[i][p] for p in valid_pos]
            true_labels_ids.append(true_seq)
            pred_labels_ids.append(np.array(pred_seq))

    from models.NER_model import ID2LABEL
    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    for true_seq, pred_seq in zip(true_labels_ids, pred_labels_ids):
        true_labels.append([ID2LABEL.get(int(t), 'O') for t in true_seq])
        pred_labels.append([ID2LABEL.get(int(p), 'O') for p in pred_seq])

    # 计算总体指标
    overall_accuracy = accuracy_score(true_labels, pred_labels)
    overall_precision = precision_score(true_labels, pred_labels) if any(len(s) for s in true_labels) else 0.0
    overall_recall = recall_score(true_labels, pred_labels) if any(len(s) for s in true_labels) else 0.0
    overall_f1 = f1_score(true_labels, pred_labels) if any(len(s) for s in true_labels) else 0.0

    report = classification_report(true_labels, pred_labels, output_dict=True) if true_labels else {}

    entity_metrics = {}
    for ent in ['PER', 'LOC', 'ORG']:
        if ent in report:
            entity_metrics[ent] = {
                'precision': float(report[ent]['precision']),
                'recall': float(report[ent]['recall']),
                'f1': float(report[ent]['f1-score']),
                'support': int(report[ent]['support'])
            }

    metrics = {
        'accuracy': float(overall_accuracy),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'entity_metrics': entity_metrics
    }
    return metrics


def format_metrics(metrics: dict, decimals: int = 4) -> str:
    """将评估结果格式化为更易读的文本表格字符串"""
    def fmt(v):
        if isinstance(v, (int,)):
            return str(v)
        try:
            return f"{float(v):.{decimals}f}"
        except Exception:
            return str(v)

    lines = []
    lines.append("总体指标:")
    lines.append(f"  Accuracy : {fmt(metrics['accuracy'])}")
    lines.append(f"  Precision: {fmt(metrics['precision'])}")
    lines.append(f"  Recall   : {fmt(metrics['recall'])}")
    lines.append(f"  F1       : {fmt(metrics['f1'])}")
    if metrics.get('entity_metrics'):
        lines.append("")
        lines.append("实体级指标 (support>=1 的标签):")
        header = f"    {'Entity':<6} {'P':>8} {'R':>8} {'F1':>8} {'Sup':>6}"
        lines.append(header)
        lines.append("    " + "-" * (len(header)-4))
        for ent, vals in metrics['entity_metrics'].items():
            lines.append(
                f"    {ent:<6} {fmt(vals['precision']):>8} {fmt(vals['recall']):>8} {fmt(vals['f1']):>8} {vals['support']:>6}"
            )
    return "\n" + "\n".join(lines)

def save_model(model, output_dir, name):
    """保存模型和配置 (带磁盘空间检查 + 原子写 + 回退策略 + 轮换清理)"""
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    # 磁盘空间检查
    try:
        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < TrainConfig.free_space_warn_gb:
            logger.warning(f"可用磁盘空间仅 {free_gb:.3f} GB, 可能导致写入失败 (阈值 {TrainConfig.free_space_warn_gb} GB)")
    except Exception as _e:  # 忽略空间检查错误
        logger.debug(f"磁盘空间检查失败: {_e}")

    state_dict = model.state_dict()
    final_path = os.path.join(model_dir, "pytorch_model.bin")
    tmp_path = final_path + ".tmp"

    def _try_save(use_legacy: bool):
        torch.save(
            state_dict,
            tmp_path,
            _use_new_zipfile_serialization=not use_legacy
        )

    save_ok = False
    first_error: Exception | None = None
    # 1) 先尝试新 zipfile 序列化 (默认)
    try:
        _try_save(use_legacy=False)
        save_ok = True
        logger.debug("使用新 zip 序列化格式保存成功")
    except Exception as e1:
        first_error = e1
        logger.warning(f"使用新序列化写入失败: {e1}")
        # 2) 可选回退旧格式
        if TrainConfig.legacy_torch_save_fallback:
            try:
                _try_save(use_legacy=True)
                save_ok = True
                logger.info("回退旧序列化格式保存成功 (_use_new_zipfile_serialization=False)")
            except Exception as e2:
                logger.error(f"旧格式保存同样失败: {e2}")
                first_error = first_error or e2

    if not save_ok:
        # 给出进一步排查建议
        logger.error(
            "模型保存失败。排查建议: 1) 检查磁盘剩余空间 df -h 2) 确认输出目录有写权限 3) 若为网络/挂载盘尝试改到本地磁盘 4) 降低checkpoint频率"
        )
        raise first_error if first_error else RuntimeError("模型保存失败且未捕获具体异常")

    # 原子替换
    try:
        os.replace(tmp_path, final_path)
    except Exception as e:
        logger.error(f"原子替换写入失败: {e}")
        raise

    # 保存配置
    try:
        with open(os.path.join(model_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(model.config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"保存 config.json 失败: {e}")

    logger.info(f"模型保存到 {model_dir}")
    _rotate_checkpoints(output_dir)
    if name.startswith("best_model_f1_"):
        _rotate_best_models(output_dir)


_CHECKPOINT_CACHE_FILE = "checkpoint_manifest.json"

def _rotate_checkpoints(output_dir: str):
    """根据配置 max_checkpoints_to_keep 保留最新若干个 checkpoint / best / final 不强制删除。
    只轮换名称以 'checkpoint-' 开头的目录。"""
    try:
        cfg = TrainConfig()  # 使用默认实例 (当前实现不在运行时动态覆盖该值, 简化)
        max_keep = cfg.max_checkpoints_to_keep
        manifest_path = os.path.join(output_dir, _CHECKPOINT_CACHE_FILE)
        # 获取所有 checkpoint-* 目录
        entries = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
        if len(entries) <= max_keep:
            return
        # 按修改时间排序 (新->旧)
        entries_full = [(d, os.path.getmtime(os.path.join(output_dir, d))) for d in entries]
        entries_full.sort(key=lambda x: x[1], reverse=True)
        to_remove = entries_full[max_keep:]
        for d, _ in to_remove:
            rm_path = os.path.join(output_dir, d)
            try:
                shutil.rmtree(rm_path)
                logger.info(f"清理旧checkpoint: {d}")
            except Exception as e:
                logger.warning(f"删除旧checkpoint失败 {d}: {e}")
        # 更新 manifest (简单记录)
        active = [d for d, _ in entries_full[:max_keep]]
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump({"active_checkpoints": active, "timestamp": time.time()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"轮换checkpoint出错: {e}")


def _rotate_best_models(output_dir: str):
    """清理 best_model_f1_* 目录, 仅保留最高 F1 的前 N 个 (按目录名解析)。"""
    try:
        cfg = TrainConfig()
        if not cfg.delete_redundant_on_save:
            return
        max_keep = max(1, cfg.max_best_models_to_keep)
        prefix = "best_model_f1_"
        entries = [d for d in os.listdir(output_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(output_dir, d))]
        if len(entries) <= max_keep:
            return
        scored = []
        for d in entries:
            # 名称格式: best_model_f1_{f1:.4f}
            try:
                f1_str = d[len(prefix):]
                score = float(f1_str)
            except Exception:
                score = -1.0  # 解析失败的放低优先级
            scored.append((d, score, os.path.getmtime(os.path.join(output_dir, d))))
        # 按 score(desc) -> mtime(desc) 排序
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        to_remove = scored[max_keep:]
        for d, score, _ in to_remove:
            path = os.path.join(output_dir, d)
            try:
                shutil.rmtree(path)
                logger.info(f"清理旧 best 模型: {d} (F1={score:.4f})")
            except Exception as e:
                logger.warning(f"删除旧 best 模型失败 {d}: {e}")
    except Exception as e:
        logger.debug(f"轮换 best 模型出错: {e}")

def main():
    """入口：支持直接运行使用内置配置，或在 USE_CLI=True 时启用命令行覆盖"""
    if USE_CLI and len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_file", type=str, default=TrainConfig.train_file)
        parser.add_argument("--eval_file", type=str, default=TrainConfig.eval_file)
        parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
        parser.add_argument("--bert_model", type=str, default=TrainConfig.bert_model)
        parser.add_argument("--freeze_bert_layers", type=int, default=TrainConfig.freeze_bert_layers,
                            help="None/0: 不冻结; -1: 全部冻结; 正整数N: 冻结前N层")
        parser.add_argument("--num_train_epochs", type=int, default=TrainConfig.num_train_epochs)
        parser.add_argument("--early_stopping", action="store_true")
        parser.add_argument("--patience", type=int, default=TrainConfig.patience)
        parser.add_argument("--bert_lr", type=float, default=TrainConfig.bert_lr)
        parser.add_argument("--other_lr", type=float, default=TrainConfig.other_lr)
        parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
        parser.add_argument("--max_seq_length", type=int, default=TrainConfig.max_seq_length)
        args = parser.parse_args()
        base = TrainConfig()
        # 合并缺省字段（理论上 parser 都有，但保持安全）
        for k, v in asdict(base).items():
            if not hasattr(args, k):
                setattr(args, k, v)
    else:
        args = build_namespace_from_dataclass(TrainConfig())

    os.makedirs(args.output_dir, exist_ok=True)
    global logger
    logger = setup_logging(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        train(args)
    except Exception as e:
        logger.exception(f"训练过程发生异常: {e}")
        raise


if __name__ == "__main__":
    main()