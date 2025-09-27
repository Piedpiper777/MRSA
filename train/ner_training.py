"""
BERT-BiLSTM-CRF 命名实体识别模型训练脚本
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import sys
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


class MSRANERDataset(Dataset):
    """MSRA NER数据集加载器"""

    def __init__(self, data_file, tokenizer, max_length=128):
        """
        初始化数据集
        
        Args:
            data_file: 数据文件路径
            tokenizer: BERT tokenizer
            max_length: 序列最大长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        """加载训练数据并预处理"""
        logger.info(f"从 {data_file} 加载数据...")

        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
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
        
        logger.info(f"加载了 {len(processed_data)} 个样本")
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
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    # 加载数据集，支持自动划分验证集
    if args.eval_file is None:
        logger.info("未指定eval_file，将自动从训练集划分10%作为验证集...")
        with open(args.train_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        from sklearn.model_selection import train_test_split
        train_data, eval_data = train_test_split(
            all_data, test_size=0.1, random_state=args.seed
        )
        # 保存划分后的验证集到临时文件
        eval_file_path = os.path.join(args.output_dir, "_auto_eval.json")
        with open(eval_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        # 保存划分后的训练集到临时文件
        train_file_path = os.path.join(args.output_dir, "_auto_train.json")
        with open(train_file_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        train_dataset = MSRANERDataset(train_file_path, tokenizer, max_length=args.max_seq_length)
        eval_dataset = MSRANERDataset(eval_file_path, tokenizer, max_length=args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    else:
        train_dataset = MSRANERDataset(args.train_file, tokenizer, max_length=args.max_seq_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        eval_dataset = MSRANERDataset(args.eval_file, tokenizer, max_length=args.max_seq_length)
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
    
    # 如果需要，冻结BERT层
    if args.freeze_bert:
        if args.freeze_bert_layers is not None:
            logger.info(f"冻结BERT前 {args.freeze_bert_layers} 层")
            model.freeze_bert_layers(args.freeze_bert_layers)
        else:
            logger.info("冻结所有BERT层")
            model.freeze_bert_layers()
        
        # 打印可训练参数信息
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"冻结后可训练参数数量: {trainable_params}")
    
    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # BERT参数 - 小学习率，有权重衰减
        {
            'params': [p for n, p in model.bert.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 2e-5,
            'weight_decay': 0.1
        },
        # BERT偏置和LayerNorm - 小学习率，无权重衰减
        {
            'params': [p for n, p in model.bert.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 2e-5,
            'weight_decay': 0.0
        },
        # LSTM参数 - 中等学习率，有权重衰减
        {
            'params': [p for n, p in model.named_parameters() 
                      if n.startswith('lstm') and not any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 1e-3,
            'weight_decay': 0.1
        },
        # LSTM偏置 - 中等学习率，无权重衰减
        {
            'params': [p for n, p in model.named_parameters() 
                      if n.startswith('lstm') and any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 1e-3,
            'weight_decay': 0.0
        },
        # 分类器和CRF参数 - 中等学习率，有权重衰减
        {
            'params': [p for n, p in model.named_parameters() 
                      if (n.startswith('classifier') or n.startswith('crf')) and not any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 1e-3,
            'weight_decay': 0.1
        },
        # 分类器偏置 - 中等学习率，无权重衰减
        {
            'params': [p for n, p in model.named_parameters() 
                      if (n.startswith('classifier') or n.startswith('crf')) and any(nd in n for nd in no_decay) and p.requires_grad],
            'lr': 1e-3,
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # 学习率调度器
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    # 训练循环
    logger.info("***** 开始训练 *****")
    global_step = 0
    best_f1 = 0
    
    # 早停相关变量
    no_improvement_count = 0
    early_stop = False
    
    for epoch in trange(args.num_train_epochs, desc="Epoch"):
        model.train()
        epoch_loss = 0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            loss, _ = model(input_ids, attention_mask, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            epoch_loss += loss.item()
            
            # 输出日志
            if step % args.logging_steps == 0:
                logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} 平均损失: {avg_epoch_loss:.4f}")
        
        # 评估
        if eval_dataloader:
            eval_metrics = evaluate(model, eval_dataloader, device)
            logger.info(f"评估结果: {eval_metrics}")
            
            current_f1 = eval_metrics['f1']
            
            # 保存最佳模型
            if current_f1 > best_f1 + args.min_delta:
                logger.info(f"模型性能有所提升！从 F1={best_f1:.4f} 提升到 {current_f1:.4f}")
                best_f1 = current_f1
                save_model(model, args.output_dir, f"best_model_f1_{best_f1:.4f}")
                no_improvement_count = 0  # 重置计数器
            else:
                no_improvement_count += 1
                logger.info(f"模型性能未有显著提升，连续 {no_improvement_count} 轮无改进")
                
            # 早停检查
            if args.early_stopping and no_improvement_count >= args.patience:
                logger.info(f"\n\n*** 早停触发！连续 {args.patience} 轮无改进 ***")
                early_stop = True
                break
        
        # 保存检查点
        if (epoch + 1) % args.save_steps == 0:
            save_model(model, args.output_dir, f"checkpoint-{epoch+1}")
            
        # 如果触发早停，跳出训练循环
        if early_stop:
            logger.info("由于早停策略触发，提前结束训练")
            break
    
    # 保存最终模型
    save_model(model, args.output_dir, "final_model")
    
    # 返回训练结果统计
    training_stats = {
        "best_f1": best_f1,
        "completed_epochs": epoch + 1,
        "early_stopped": early_stop,
        "global_steps": global_step
    }
    
    logger.info(f"训练统计信息: {training_stats}")
    logger.info("训练完成！")


def evaluate( model, eval_dataloader, device):
    """评估模型"""
    model.eval()
    
    true_labels_ids = []
    pred_labels_ids = []
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # 准备数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        # 预测
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            predictions = model.decode(logits, attention_mask)
        
        # 收集预测结果和真实标签
        for i, mask in enumerate(attention_mask):
            true_seq = labels[i][labels[i] != -100].cpu().numpy()
            pred_seq = np.array(predictions[i])[:len(true_seq)]
            
            true_labels_ids.append(true_seq)
            pred_labels_ids.append(pred_seq)
    
    # 转换为标签文本以便使用seqeval进行评估
    true_labels = []
    pred_labels = []
    from models.NER_model import ID2LABEL
    
    for true_seq, pred_seq in zip(true_labels_ids, pred_labels_ids):
        true_label_text = [ID2LABEL.get(label_id, "O") for label_id in true_seq]
        pred_label_text = [ID2LABEL.get(label_id, "O") for label_id in pred_seq]
        
        true_labels.append(true_label_text)
        pred_labels.append(pred_label_text)
    
    # 使用seqeval计算评估指标
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels)
    }
    
    # 获取详细的分类报告
    report = classification_report(true_labels, pred_labels, output_dict=True)
    metrics['entity_metrics'] = {}
    
    # 提取每个实体类型的指标
    for entity_type in ['PER', 'LOC', 'ORG']:
        # seqeval使用完整标签，如'B-PER'而不仅仅是'PER'
        b_tag = f'B-{entity_type}'
        if b_tag in report:
            metrics['entity_metrics'][entity_type] = {
                'precision': report[b_tag]['precision'],
                'recall': report[b_tag]['recall'],
                'f1': report[b_tag]['f1-score'],
                'support': report[b_tag]['support']
            }
    
    return metrics

def save_model(model, output_dir, name):
    """保存模型和配置"""
    model_dir = os.path.join(output_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型参数
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    
    # 保存模型配置
    with open(os.path.join(model_dir, "config.json"), 'w') as f:
        json.dump(model.config, f)
    
    logger.info(f"模型保存到 {model_dir}")

def main():
    # 异常处理和日志记录
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--train_file", default="../data/train_data/train.json", type=str,
                        help="训练数据文件路径")
    parser.add_argument("--eval_file", default=None, type=str,
                        help="验证数据文件路径")
    parser.add_argument("--output_dir", default="../models/saved_models", type=str,
                        help="模型输出目录")
    
    # 模型参数
    parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                        help="预训练BERT模型名称或路径")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="输入序列的最大长度")
    parser.add_argument("--lstm_hidden_size", default=256, type=int,
                        help="LSTM隐藏层大小")
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="LSTM层数")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="Dropout比率")
    
    # 训练参数
    parser.add_argument("--freeze_bert", action="store_true",
                        help="是否冻结BERT层的参数")
    parser.add_argument("--freeze_bert_layers", default=None, type=int,
                        help="要冻结的BERT层数，如果不设置但--freeze_bert为True，则冻结所有层")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="训练批次大小")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="初始学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam优化器epsilon值")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="梯度裁剪阈值")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="训练轮数")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="学习率预热步数")
    parser.add_argument("--logging_steps", default=100, type=int,
                        help="日志输出间隔步数")
    parser.add_argument("--save_steps", default=1, type=int,
                        help="保存检查点的轮数间隔")
    parser.add_argument("--seed", default=42, type=int,
                        help="随机种子")
    
    # 早停相关参数
    parser.add_argument("--patience", default=3, type=int,
                        help="早停耐心值，默认3个轮次不再提升则停止训练")
    parser.add_argument("--min_delta", default=0.0001, type=float,
                        help="早停最小变化阈值，指标变化小于该值不计入改进")
    parser.add_argument("--early_stopping", action="store_true",
                        help="是否启用早停机制")
                        
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建日志记录
    global logger
    logger = setup_logging(args)
    
    try:
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
        # 检查依赖库
        try:
            import sklearn
            import seqeval
        except ImportError as e:
            logger.error(f"缺少依赖库: {e}")
            logger.info("请安装缺失的库: pip install scikit-learn seqeval")
            return
    
        # 开始训练
        train(args)
    except Exception as e:
        logger.exception(f"训练过程发生异常: {e}")
        raise


if __name__ == "__main__":
    main()