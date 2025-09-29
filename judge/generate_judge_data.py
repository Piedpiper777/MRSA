"""
K折交叉验证生成Judge模型训练数据

思路:
1. 将Gold标注数据分为验证集(10%)和训练集(90%)
2. 对训练集进行K折划分
3. 每一折: 用其余k-1折训练NER模型，预测当前折生成judge样本
4. 合并所有折的预测结果作为judge训练数据

Judge样本特征:
- 原始tokens和金标准标签
- 模型预测标签  
- 实体级别的准确性标记(正确/错误)
- 置信度分数(可选)
- 预测与真实的差异统计
"""

import os
import json
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Tuple
import torch
from dataclasses import dataclass
from tqdm import tqdm
import sys
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.ner_training import TrainConfig, build_namespace_from_dataclass, train
from predict.ner_predict import load_model, predict_one, load_custom_config
from models.NER_model import ID2LABEL
from transformers import AutoTokenizer
import shutil

@dataclass
class JudgeDataConfig:
    """Judge数据生成配置"""
    # 输入数据路径
    gold_data_path: str = r"/workspace/data/train_data/splits/1-99/labeled_1%.json" #Gold标注数据
    
    # 输出路径
    judge_data_dir: str = r"/workspace/data/judge_data/1-99" #judge数据输出目录
    val_data_path: str = r"/workspace/data/judge_data/1-99/validation.json" #验证集保存路径
    
    # K折设置
    k_folds: int = 40
    random_seed: int = 42
    val_ratio: float = 0.1
    
    # 输出控制
    save_models: bool = False  # 是否保存中间NER模型
    verbose: bool = True


def create_train_config(judge_config: JudgeDataConfig) -> TrainConfig:
    """为K折训练创建优化的TrainConfig"""
    return TrainConfig(
        # 训练文件路径（会在每折时动态设置）
        train_file="",  # 动态设置
        eval_file="",   # 动态设置
        output_dir="",  # 动态设置
        
        # 模型配置
        bert_model=r"/workspace/models/google-bert/bert-base-chinese",
        max_seq_length=128,
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.1,
        freeze_bert_layers=None,  # None/0: 不冻结; -1: 全部冻结; 正整数N: 冻结前N层
        
        # 训练参数
        batch_size=16,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=30,
        warmup_steps=0,
        
        # 分组学习率
        bert_lr=2e-5,
        other_lr=1e-3,
        bert_weight_decay=0.1,
        other_weight_decay=0.1,
        
        # 早停和验证
        patience=5,
        min_delta=0.0001,
        early_stopping=True,
        logging_steps=100,
        save_steps=1,
        
        # Checkpoint管理(为K折训练优化)
        max_checkpoints_to_keep=1,  # 节省空间
        max_best_models_to_keep=1,  # 每折只保留最好的模型
        delete_redundant_on_save=True,  # 立即清理旧模型
        
        # 随机种子
        seed=judge_config.random_seed
    )


def load_gold_data(data_path: str) -> List[Dict[str, Any]]:
    """加载金标准数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    else:
        # 兼容其他格式
        return data.get('data', [])


def split_train_val(data: List[Dict], val_ratio: float, random_seed: int) -> Tuple[List[Dict], List[Dict]]:
    """
    划分训练集和验证集
    Args:
        data: 完整数据列表
        val_ratio: 验证集比例
        random_seed: 随机种子
    returns: (train_data, val_data)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(data))
    val_size = int(len(data) * val_ratio)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    
    return train_data, val_data


def create_k_folds(data: List[Dict], k: int, random_seed: int) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    创建K折数据划分，得到k-1折训练和1折测试
    Args:
        data: 训练数据列表
        k: 折数
        random_seed: 随机种子
    returns: List of (train_fold, test_fold) tuples
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
    
    folds = []
    for train_idx, test_idx in kf.split(np.arange(len(data))):
        fold_train = [data[i] for i in train_idx]
        fold_test = [data[i] for i in test_idx]
        folds.append((fold_train, fold_test))
    
    return folds


def prepare_fold_data(fold_train: List[Dict], fold_id: int, temp_dir: str) -> str:
    """
    为当前折准备训练数据文件，把列表写入json文件
    Args:
        fold_train: 当前折的训练数据列表
        fold_id: 折编号
        temp_dir: 临时目录
    returns: k-1折训练数据文件路径
    """
    fold_index = fold_id + 1
    fold_train_path = os.path.join(temp_dir, f"fold_{fold_index}_train.json")#把训练数据写到临时文件，并且命名为fold_{fold_index}_train.json
    #此时的fold_train_path是一个文件夹路径
    with open(fold_train_path, 'w', encoding='utf-8') as f:
        json.dump(fold_train, f, ensure_ascii=False, indent=2)
        #打开训练数据文件，将训练数据以json格式写入文件
        #ensure_ascii=False保证中文不被转义，indent=2美化格式
        #最后得到一个包含当前折训练数据的json文件
    
    return fold_train_path #此时的fold_train_path是一个json文件路径


def train_fold_model(fold_train_path: str, fold_id: int, judge_config: JudgeDataConfig, temp_dir: str) -> str:
    """
    训练当前折的NER模型，复用ner_training.py的训练逻辑
    Args:
        fold_train_path: 当前折训练数据文件路径(JSON文件)
        fold_id: 折编号
        judge_config: Judge数据生成配置
        temp_dir: 临时目录
    returns: 最佳模型路径
    """
    fold_index = fold_id + 1
    model_dir = os.path.join(temp_dir, f"fold_{fold_index}_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建TrainConfig实例
    train_config = create_train_config(judge_config)
    
    # 设置fold特定的参数
    train_config.train_file = fold_train_path  # 当前折训练数据
    train_config.eval_file = judge_config.val_data_path  # 验证集，固定不变
    train_config.output_dir = model_dir  # 模型输出目录
    
    # 转换为namespace格式（train函数需要）
    args = build_namespace_from_dataclass(train_config)
    
    if judge_config.verbose:
        print(f"训练第{fold_id+1}折模型...")
        print(f"配置: max_checkpoints={args.max_checkpoints_to_keep}, max_best_models={args.max_best_models_to_keep}")
        print(f"删除冗余模型: {args.delete_redundant_on_save}")
    
    # 调用ner_training.py中的train函数
    train(args)
    
    # 查找生成的最佳模型路径
    # ner_training.py会生成类似best_model_f1_0.xxxx的目录
    best_model_pattern = os.path.join(model_dir, "best_*")#从model_dir目录下查找所有以best_开头的文件或文件夹
    best_models = glob.glob(best_model_pattern)#glob.glob返回所有匹配的文件路径列表
    
    if not best_models:
        raise FileNotFoundError(f"未找到训练生成的最佳模型，检查目录: {model_dir}")
    
    # 选择最新的最佳模型（按修改时间）
    best_model_path = max(best_models, key=os.path.getmtime)
    
    # # 强制清理多余的模型文件（额外保障）
    if len(best_models) > 1:
        for model_path in best_models:
            if model_path != best_model_path:
                try:
                    shutil.rmtree(model_path)
                    if judge_config.verbose:
                        print(f"清理多余的best模型: {model_path}")
                except Exception as e:
                    print(f"清理多余模型失败: {e}")
    
    # 清理多余的checkpoint文件
    checkpoint_pattern = os.path.join(model_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    if len(checkpoints) > train_config.max_checkpoints_to_keep:
        # 按修改时间排序，保留最新的
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        for checkpoint in checkpoints[train_config.max_checkpoints_to_keep:]:
            try:
                shutil.rmtree(checkpoint)
                if judge_config.verbose:
                    print(f"清理多余的checkpoint: {checkpoint}")
            except Exception as e:
                print(f"清理checkpoint失败: {e}")
    
    if judge_config.verbose:
        print(f"第{fold_id+1}折模型训练完成: {best_model_path}")
    
    return best_model_path


def extract_entity_level_features(tokens: List[str], true_tags: List[str], pred_tags: List[str]) -> List[Dict[str, Any]]:
    """
    提取实体级别的judge特征
    Args:
        tokens: 词列表
        true_tags: 金标准BIO标签列表
        pred_tags: 预测BIO标签列表
    returns: List of entity-level feature dicts
    """
    
    def extract_entities(tags):
        """
        从BIO标签提取实体
        Args:
            tags: BIO标签列表
        returns: List of entities with start, end, type
        例如: [{'type': 'PER', 'start': 0, 'end': 2}, ...]
        """
        entities = []
        current = None
        for idx, tag in enumerate(tags):
            if tag.startswith('B-'):
                #如果标签是B-开头，表示一个新实体的开始
                if current:
                    #如果当前已有实体
                    entities.append(current)
                    #把当前的实体加入实体列表entities
                current = {'type': tag[2:], 'start': idx, 'end': idx}
                #重新开始一个新实体，记录类型和起始位置，类型是标签去掉B-的部分
            elif tag.startswith('I-') and current and current['type'] == tag[2:]:
                #如果标签是I-开头，且当前已有实体，且类型匹配
                current['end'] = idx
                #更新当前实体的结束位置，不断循环，直到最后一个I-标签，就是实体的结束位置
            else:
                #遇到O标签或不匹配的I-标签，表示实体结束
                if current:
                    #如果当前已有实体
                    entities.append(current)
                    #把当前实体加入实体列表entities
                    current = None
                    #重置当前实体为None，等待下一个B-标签开始新实体
        if current:
            #如果当前还有未加入的实体
            entities.append(current)
            #把最后一个实体加入实体列表entities
        return entities
    
    def create_entity_feature(entity: Dict, tokens: List[str], is_correct: bool, match_type: str) -> Dict[str, Any]:
        """
        创建实体特征字典
        Args:
            entity: 实体字典，包含type, start, end
            tokens: 词列表
            is_correct: 实体是否正确匹配
            match_type: 匹配类型描述
        """
        start, end = entity['start'], entity['end']
        return {
            'entity_text': ''.join(tokens[start:end+1]),#根据实体的起始和结束位置，从tokens中提取实体文本
            'entity_type': entity['type'],
            'start_pos': start,
            'end_pos': end,
            'length': end - start + 1,
            'is_correct': is_correct,
            'match_type': match_type,
            'tokens': tokens[start:end+1],
            'context_before': tokens[max(0, start-2):start] if start > 0 else [],#取实体前两个词作为上下文
            'context_after': tokens[end+1:min(len(tokens), end+3)] if end < len(tokens)-1 else [] #取实体后两个词作为上下文
        }
    
    true_entities = extract_entities(true_tags)
    #从真实标签中提取实体列表，例如: [{'type': 'PER', 'start': 0, 'end': 2}, ...]
    pred_entities = extract_entities(pred_tags)#从预测标签中提取实体列表
    
    # 创建真实实体的快速查找集合，例如: {(0, 2, 'PER'), ...}
    true_entity_set = {(ent['start'], ent['end'], ent['type']) for ent in true_entities}
    
    features = []
    processed_true_entities = set()
    
    # 处理预测实体
    for pred_ent in pred_entities:
        pred_key = (pred_ent['start'], pred_ent['end'], pred_ent['type'])
        #例如: (0, 2, 'PER')
        
        if pred_key in true_entity_set:
            # 完全匹配
            features.append(create_entity_feature(pred_ent, tokens, True, "exact"))
            processed_true_entities.add(pred_key)
        else:
            # 检查部分匹配
            match_type = "no_match"
            for true_ent in true_entities:
                if (pred_ent['start'] == true_ent['start'] and pred_ent['end'] == true_ent['end']):
                    #如果起始和结束位置都匹配，但类型不匹配
                    match_type = "boundary_correct_type_wrong"
                    break
                elif pred_ent['type'] == true_ent['type']:
                    #如果类型匹配，但位置不匹配
                    match_type = "type_correct_boundary_wrong"
                else:
                    #完全不匹配
                    match_type = "type_wrong_boundary_wrong"
            
            features.append(create_entity_feature(pred_ent, tokens, False, match_type))
    
    # 处理遗漏的真实实体
    for true_ent in true_entities:
        true_key = (true_ent['start'], true_ent['end'], true_ent['type'])
        if true_key not in processed_true_entities:
            features.append(create_entity_feature(true_ent, tokens, False, "missed_entity"))
    
    return features


def generate_judge_samples(fold_test: List[Dict], model_path: str, judge_config: JudgeDataConfig) -> List[Dict[str, Any]]:
    """
    使用训练好的模型预测测试fold，生成judge样本
    Args:
        fold_test: 当前折的测试数据列表
        model_path: 训练好的模型路径
        judge_config: Judge数据生成配置
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和tokenizer
    cfg = load_custom_config(model_path)
    # 使用TrainConfig中的bert_model路径
    train_config = create_train_config(judge_config)
    tokenizer = AutoTokenizer.from_pretrained(train_config.bert_model, use_fast=True)
    model = load_model(model_path, train_config.bert_model, device)#加载模型
    max_length = cfg.get('max_length', 128)
    
    judge_samples = []
    
    for sample in tqdm(fold_test, desc="生成judge样本"):
        tokens = sample['tokens']
        true_tags = sample['labels']
        
        # 模型预测
        pred_result = predict_one(model, tokenizer, tokens, device, max_length)
        pred_tags = pred_result['tags']
        
        # 确保长度一致
        min_len = min(len(tokens), len(true_tags), len(pred_tags))
        tokens = tokens[:min_len]
        true_tags = true_tags[:min_len]
        pred_tags = pred_tags[:min_len]
        
        # 提取实体级别特征
        entity_features = extract_entity_level_features(tokens, true_tags, pred_tags)
        
        # 计算序列级别统计
        token_accuracy = sum(1 for t, p in zip(true_tags, pred_tags) if t == p) / len(true_tags)
        
        # 计算实体统计
        pred_entities = [f for f in entity_features if f['match_type'] != 'missed_entity']#预测的实体数
        true_entities = len(entity_features)  # 真实的实体数，包括预测的和遗漏的
        correct_entities = len([f for f in entity_features if f['is_correct']])#正确的实体数
        
        # 构造judge样本
        judge_sample = {
            'id': sample.get('id', f"sample_{len(judge_samples)}"),
            'tokens': tokens,
            'true_labels': true_tags,
            'pred_labels': pred_tags,
            'entity_features': entity_features,
            'sequence_accuracy': token_accuracy,
            'sequence_length': len(tokens),
            'num_entities_pred': len(pred_entities),
            'num_entities_total': true_entities,
            'num_correct_entities': correct_entities
        }
        
        judge_samples.append(judge_sample)
    
    return judge_samples


def main():
    config = JudgeDataConfig()
    summary_train_config = create_train_config(config)
    
    print("=== K折Judge数据生成开始 ===")
    print(f"配置: K={config.k_folds}, 验证集比例={config.val_ratio}")
    
    # 创建输出目录
    os.makedirs(config.judge_data_dir, exist_ok=True)
    temp_dir = os.path.join(config.judge_data_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 加载金标准数据
    print("加载金标准数据...")
    gold_data = load_gold_data(config.gold_data_path)
    print(f"总样本数: {len(gold_data)}")
    
    # 划分训练集和验证集
    train_data, val_data = split_train_val(gold_data, config.val_ratio, config.random_seed)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 保存验证集
    os.makedirs(os.path.dirname(config.val_data_path), exist_ok=True)
    with open(config.val_data_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    # 创建K折
    print(f"创建{config.k_folds}折数据...")
    folds = create_k_folds(train_data, config.k_folds, config.random_seed)
    
    all_judge_samples = []
    
    # 处理每一折
    for fold_id, (fold_train, fold_test) in enumerate(folds):
        print(f"\n--- 处理第{fold_id+1}/{config.k_folds}折 ---")
        print(f"训练样本: {len(fold_train)}, 测试样本: {len(fold_test)}")
        
        # 准备训练数据
        fold_train_path = prepare_fold_data(fold_train, fold_id, temp_dir)
        
        # 训练模型（只保留最优模型）
        model_path = train_fold_model(fold_train_path, fold_id, config, temp_dir)
        
        # 使用最优模型预测fold_test，生成judge样本
        fold_judge_samples = generate_judge_samples(fold_test, model_path, config)
        all_judge_samples.extend(fold_judge_samples)
        
        print(f"第{fold_id+1}折生成judge样本: {len(fold_judge_samples)}")
        
        # 保存当前折的预测结果和统计信息
        fold_results_dir = os.path.join(config.judge_data_dir, f"fold_{fold_id+1}_results")
        os.makedirs(fold_results_dir, exist_ok=True)
        
        # 保存当前折的judge样本
        fold_judge_path = os.path.join(fold_results_dir, f"fold_{fold_id+1}_judge_samples.json")
        with open(fold_judge_path, 'w', encoding='utf-8') as f:
            json.dump(fold_judge_samples, f, ensure_ascii=False, indent=2)
        
        # 计算并保存当前折的统计信息
        fold_total_entities = sum(sample['num_entities_total'] for sample in fold_judge_samples)
        fold_correct_entities = sum(sample['num_correct_entities'] for sample in fold_judge_samples)
        fold_accuracy = fold_correct_entities / fold_total_entities if fold_total_entities > 0 else 0
        
        fold_stats = {
            "fold_id": fold_id + 1,
            "model_path": model_path,
            "train_samples": len(fold_train),
            "test_samples": len(fold_test),
            "judge_samples": len(fold_judge_samples),
            "total_entities": fold_total_entities,
            "correct_entities": fold_correct_entities,
            "entity_accuracy": fold_accuracy,
            "avg_sequence_accuracy": sum(sample['sequence_accuracy'] for sample in fold_judge_samples) / len(fold_judge_samples) if fold_judge_samples else 0
        }
        
        fold_stats_path = os.path.join(fold_results_dir, f"fold_{fold_id+1}_stats.json")
        with open(fold_stats_path, 'w', encoding='utf-8') as f:
            json.dump(fold_stats, f, ensure_ascii=False, indent=2)
        
        if config.verbose:
            print(f"第{fold_id+1}折结果已保存到: {fold_results_dir}")
            print(f"实体准确率: {fold_accuracy:.3f}, 序列准确率: {fold_stats['avg_sequence_accuracy']:.3f}")
        
        # 可选择性保存模型
        if config.save_models:
            # 复制模型到永久位置
            permanent_model_path = os.path.join(fold_results_dir, "model")
            try:
                shutil.copytree(model_path, permanent_model_path, dirs_exist_ok=True)
                if config.verbose:
                    print(f"第{fold_id+1}折模型已保存到: {permanent_model_path}")
            except Exception as e:
                print(f"保存第{fold_id+1}折模型失败: {e}")
        
        # 清理当前折的临时模型（已完成预测且已保存结果）
        if not config.save_models:
            model_dir = os.path.dirname(model_path)
            if os.path.exists(model_dir):
                try:
                    shutil.rmtree(model_dir)
                    if config.verbose:
                        print(f"已清理第{fold_id+1}折临时模型: {model_dir}")
                except Exception as e:
                    print(f"清理第{fold_id+1}折模型失败: {e}")
    
    # 保存最终合并的judge数据
    judge_output_path = os.path.join(config.judge_data_dir, "judge_training_data.json")
    with open(judge_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_judge_samples, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    total_entities = sum(sample['num_entities_total'] for sample in all_judge_samples)
    correct_entities = sum(sample['num_correct_entities'] for sample in all_judge_samples)
    avg_sequence_acc = sum(sample['sequence_accuracy'] for sample in all_judge_samples) / len(all_judge_samples) if all_judge_samples else 0
    
    # 保存整体统计信息
    overall_stats = {
        "experiment_config": {
            "k_folds": config.k_folds,
            "val_ratio": config.val_ratio,
            "num_train_epochs": summary_train_config.num_train_epochs,
            "batch_size": summary_train_config.batch_size,
            "bert_lr": summary_train_config.bert_lr,
            "other_lr": summary_train_config.other_lr
        },
        "results": {
            "total_samples": len(all_judge_samples),
            "total_entities": total_entities,
            "correct_entities": correct_entities,
            "overall_entity_accuracy": correct_entities / total_entities if total_entities > 0 else 0,
            "overall_sequence_accuracy": avg_sequence_acc
        },
        "files": {
            "judge_training_data": "judge_training_data.json",
            "fold_results": [f"fold_{i+1}_results/" for i in range(config.k_folds)]
        }
    }
    
    stats_output_path = os.path.join(config.judge_data_dir, "experiment_summary.json")
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Judge数据生成完成 ===")
    print(f"总样本数: {len(all_judge_samples)}")
    print(f"总实体数: {total_entities}")
    print(f"正确实体数: {correct_entities}")
    if total_entities > 0:
        print(f"实体准确率: {correct_entities/total_entities:.3f}")
    else:
        print("实体准确率: N/A (无实体)")
    print(f"序列准确率: {avg_sequence_acc:.3f}")
    print(f"\n输出文件:")
    print(f"  - 合并judge数据: {judge_output_path}")
    print(f"  - 实验汇总: {stats_output_path}")
    print(f"  - 各折详细结果: {config.judge_data_dir}/fold_*_results/")
    
    # 清理临时目录
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
    except Exception as e:
        print(f"清理临时目录失败: {e}")


if __name__ == "__main__":
    main()