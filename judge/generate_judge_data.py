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
    gold_data_path: str = r"/workspace/MRSA/data/train_data/splits/1-99/labeled_1%.json" #Gold标注数据
    
    # 输出路径
    judge_data_dir: str = r"/workspace/MRSA/data/judge_data/1-99" #judge数据输出目录
    val_data_path: str = r"/workspace/MRSA/data/judge_data/1-99/validation.json" #验证集保存路径
    
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
        bert_model=r"/workspace/MRSA/models/google-bert/bert-base-chinese",
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
        patience=10,
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


def validate_data_format(sample: Dict[str, Any], sample_idx: int) -> bool:
    """
    验证数据格式是否正确
    Args:
        sample: 单个样本
        sample_idx: 样本索引（用于错误报告）
    Returns:
        bool: 格式是否正确
    """
    required_fields = ['tokens', 'labels']
    
    for field in required_fields:
        if field not in sample:
            print(f"错误：样本 {sample_idx} 缺少字段 '{field}'")
            return False
    
    tokens = sample['tokens']
    labels = sample['labels']
    
    # 检查数据类型
    if not isinstance(tokens, list) or not isinstance(labels, list):
        print(f"错误：样本 {sample_idx} 的 tokens 和 labels 必须是列表")
        return False
    
    # 检查长度一致性
    if len(tokens) != len(labels):
        print(f"错误：样本 {sample_idx} 的 tokens 长度({len(tokens)}) 与 labels 长度({len(labels)}) 不一致")
        return False
    
    # 检查是否为空
    if len(tokens) == 0:
        print(f"警告：样本 {sample_idx} 为空序列")
        return False
    
    # 检查BIO标签格式
    valid_prefixes = {'B-', 'I-', 'O'}
    for i, label in enumerate(labels):
        if label != 'O' and not any(label.startswith(prefix) for prefix in valid_prefixes if prefix != 'O'):
            print(f"错误：样本 {sample_idx} 位置 {i} 的标签 '{label}' 格式不正确")
            return False
    
    return True


def load_gold_data(data_path: str) -> List[Dict[str, Any]]:
    """加载金标准数据并验证格式"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        raw_data = data
    else:
        # 兼容其他格式
        raw_data = data.get('data', [])
    
    # 验证数据格式
    valid_data = []
    invalid_count = 0
    
    for i, sample in enumerate(raw_data):
        if validate_data_format(sample, i):
            valid_data.append(sample)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"警告：发现 {invalid_count} 个无效样本，已跳过")
        print(f"有效样本数：{len(valid_data)}")
    
    return valid_data


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


def extract_entity_level_features(tokens: List[str], true_tags: List[str], pred_tags: List[str], 
                                  model=None, tokenizer=None, device=None, max_length=128) -> List[Dict[str, Any]]:
    """
    提取实体级别的judge特征
    Args:
        tokens: 词列表
        true_tags: 金标准BIO标签列表
        pred_tags: 预测BIO标签列表
        model: NER模型（用于提取embeddings和logits）
        tokenizer: 分词器
        device: 设备
        max_length: 最大序列长度
    returns: List of entity-level feature dicts
    """
    
    def create_entity_feature(pred_entity: Dict, true_entity: Dict, tokens: List[str], 
                             match_type: str, span_features: Dict = None) -> Dict[str, Any]:
        """
        创建实体特征字典
        Args:
            pred_entity: 预测实体字典
            true_entity: 真实实体字典  
            tokens: 词列表
            match_type: 匹配类型描述
            span_features: 从模型提取的span特征
        """
        if pred_entity:
            pred_start = pred_entity['start']
            pred_end = pred_entity['end']
        else:
            pred_start = pred_end = None

        if true_entity:
            true_start = true_entity['start']
            true_end = true_entity['end']
        else:
            true_start = true_end = None

        if pred_entity is not None:
            main_start, main_end = pred_start, pred_end
        elif true_entity is not None:
            main_start, main_end = true_start, true_end
        else:
            main_start = main_end = None

        if main_start is not None and main_end is not None:
            # 确保索引在有效范围内
            start_idx = max(0, min(main_start, len(tokens)-1))
            end_idx = max(start_idx, min(main_end, len(tokens)-1))
            
            entity_tokens = tokens[start_idx:end_idx+1]
            entity_span = ''.join(entity_tokens)
            
            # 安全的上下文提取
            context_before = tokens[max(0, start_idx-2):start_idx] if start_idx > 0 else ["<BOS>"]
            context_after = tokens[end_idx+1:min(len(tokens), end_idx+3)] if end_idx < len(tokens)-1 else ["<EOS>"]
        else:
            entity_tokens = []
            entity_span = ""
            context_before = ["<BOS>"]
            context_after = ["<EOS>"]
        
        feature = {
            'entity_span': entity_span,
            'tokens': entity_tokens,
            'pred_start_end': [pred_start, pred_end] if pred_start is not None else None,
            'true_start_end': [true_start, true_end] if true_start is not None else None,
            'pred_type': pred_entity['type'] if pred_entity else None,
            'true_type': true_entity['type'] if true_entity else None,
            'match_type': match_type,
            'context_before': context_before,
            'context_after': context_after
        }
        
        # 添加模型特征
        if span_features:
            feature.update(span_features)
        
        return feature
    
    true_entities = extract_entities(true_tags)
    #从真实标签中提取实体列表，例如: [{'type': 'PER', 'start': 0, 'end': 2}, ...]
    pred_entities = extract_entities(pred_tags)#从预测标签中提取实体列表
    
    # 创建真实实体的快速查找集合，例如: {(0, 2, 'PER'), ...}
    true_entity_set = {(ent['start'], ent['end'], ent['type']) for ent in true_entities}
    
    features = []
    processed_true_entities = set()
    
    # 获取span特征
    span_features_cache = {}
    if model is not None and tokenizer is not None:
        span_features_cache = extract_span_features(tokens, model, tokenizer, device, max_length)
    
    # 处理预测实体
    for pred_ent in pred_entities:
        pred_key = (pred_ent['start'], pred_ent['end'], pred_ent['type'])
        
        # 查找对应的真实实体
        matched_true_ent = None
        match_type = "botherror"  # 默认为类型边界均错误
        
        if pred_key in true_entity_set:
            # 完全匹配：类型和边界均正确
            match_type = "correct"
            # 找到对应的真实实体
            matched_true_ent = next(ent for ent in true_entities if 
                                  (ent['start'], ent['end'], ent['type']) == pred_key)
            processed_true_entities.add(pred_key)
        else:
            # 检查是否为幻觉错误：预测位置的真实标签全为O
            pred_start, pred_end = pred_ent['start'], pred_ent['end']
            is_hallucination = all(tag == 'O' for tag in true_tags[pred_start:pred_end+1])
            
            if is_hallucination:
                match_type = "hallucinationerror"
                matched_true_ent = None
            else:
                # 检查部分匹配
                exact_boundary_match = False
                exact_type_match = False
                
                for true_ent in true_entities:
                    if (pred_ent['start'] == true_ent['start'] and pred_ent['end'] == true_ent['end']):
                        # 边界完全匹配
                        if pred_ent['type'] == true_ent['type']:
                            # 不应该到这里，因为会被完全匹配捕获
                            match_type = "correct"
                        else:
                            # 边界正确，类型错误
                            match_type = "typeerror"
                        matched_true_ent = true_ent
                        exact_boundary_match = True
                        processed_true_entities.add((true_ent['start'], true_ent['end'], true_ent['type']))
                        break
                
                # 如果没有找到边界完全匹配，检查类型匹配（但不应该随便匹配第一个相同类型的实体）
                if not exact_boundary_match:
                    # 寻找类型匹配且有位置重叠的实体
                    type_overlap_candidates = []
                    for true_ent in true_entities:
                        if pred_ent['type'] == true_ent['type']:
                            # 检查是否有位置重叠
                            overlap_start = max(pred_ent['start'], true_ent['start'])
                            overlap_end = min(pred_ent['end'], true_ent['end'])
                            if overlap_start <= overlap_end:  # 有实际重叠
                                overlap_len = overlap_end - overlap_start + 1
                                true_key = (true_ent['start'], true_ent['end'], true_ent['type'])
                                is_used = true_key in processed_true_entities
                                type_overlap_candidates.append((is_used, overlap_len, true_ent, true_key))
                    
                    # 如果找到类型匹配且有重叠的实体
                    if type_overlap_candidates:
                        type_overlap_candidates.sort(key=lambda x: (x[0], -x[1]))
                        best_true_ent = type_overlap_candidates[0][2]
                        best_true_key = type_overlap_candidates[0][3]
                        match_type = "boundaryerror"
                        matched_true_ent = best_true_ent
                        exact_type_match = True
                        processed_true_entities.add(best_true_key)
                
                # 如果既没有边界匹配也没有类型匹配，检查是否有真正的位置重叠
                if not exact_boundary_match and not exact_type_match:
                    overlap_candidates = []
                    for true_ent in true_entities:
                        true_key = (true_ent['start'], true_ent['end'], true_ent['type'])
                        # 检查实际的位置重叠
                        overlap_start = max(pred_ent['start'], true_ent['start'])
                        overlap_end = min(pred_ent['end'], true_ent['end'])
                        # 只有真正有重叠的位置才考虑匹配
                        if overlap_start <= overlap_end:
                            overlap_len = overlap_end - overlap_start + 1
                            is_used = true_key in processed_true_entities
                            overlap_candidates.append((is_used, overlap_len, true_ent, true_key))

                    # 只有存在真实重叠时才进行匹配
                    if overlap_candidates:
                        # 优先选择尚未匹配的实体，其次选择重叠最长的
                        overlap_candidates.sort(key=lambda x: (x[0], -x[1]))
                        best_true_ent = overlap_candidates[0][2]
                        best_true_key = overlap_candidates[0][3]
                        matched_true_ent = best_true_ent
                        processed_true_entities.add(best_true_key)
        
        # 获取span特征
        span_feats = span_features_cache.get((pred_ent['start'], pred_ent['end']), {})
        
        features.append(create_entity_feature(pred_ent, matched_true_ent, tokens, match_type, span_feats))
    
    # 处理遗漏的真实实体
    for true_ent in true_entities:
        true_key = (true_ent['start'], true_ent['end'], true_ent['type'])
        if true_key not in processed_true_entities:
            # 获取span特征
            span_feats = span_features_cache.get((true_ent['start'], true_ent['end']), {})
            features.append(create_entity_feature(None, true_ent, tokens, "missed", span_feats))
    
    return features


def extract_span_features(tokens: List[str], model, tokenizer, device, max_length: int) -> Dict[Tuple[int, int], Dict]:
    """
    提取所有可能span的特征
    Args:
        tokens: 词列表（每个汉字作为一个token）
        model: NER模型
        tokenizer: 分词器 
        device: 设备
        max_length: 最大长度
    Returns:
        Dict mapping (start, end) -> span features
    """
    try:
        # 输入验证
        if not tokens or len(tokens) == 0:
            return {}
        
        # 编码输入
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        word_ids = encoded.word_ids(batch_index=0)
        
        if word_ids is None:
            print("警告：tokenizer未返回word_ids")
            return {}

        # 仅保留模型需要的字段，避免传入不支持的 token_type_ids 等参数
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        
        # 模型前向传播
        model.eval()  # 确保模型处于评估模式
        with torch.no_grad():
            try:
                bert_outputs = model.bert(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True
                )
                sequence_output = bert_outputs.last_hidden_state
                
                # 检查LSTM层是否存在
                if hasattr(model, 'lstm') and model.lstm is not None:
                    lstm_output, _ = model.lstm(sequence_output)
                    if hasattr(model, 'dropout') and model.dropout is not None:
                        lstm_output = model.dropout(lstm_output)
                    hidden_states = lstm_output
                else:
                    # 如果没有LSTM层，使用BERT输出
                    hidden_states = sequence_output
                
                # 检查分类器是否存在
                if hasattr(model, 'classifier') and model.classifier is not None:
                    logits = model.classifier(hidden_states)
                else:
                    print("警告：模型没有classifier层，无法计算logits")
                    logits = None
                    
            except Exception as e:
                print(f"模型前向传播失败: {e}")
                return {}

        # 构建token到wordpiece的映射
        token_to_wordpiece: Dict[int, List[int]] = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            token_to_wordpiece.setdefault(word_id, []).append(idx)

        # 验证映射覆盖所有tokens
        missing_tokens = [i for i in range(len(tokens)) if i not in token_to_wordpiece]
        if missing_tokens:
            print(f"警告：以下tokens没有wordpiece映射: {missing_tokens}")

        # 为每个可能的span计算特征
        span_features = {}
        max_entity_length = min(10, len(tokens))  # 限制最大实体长度

        for start in range(len(tokens)):
            for end in range(start, min(start + max_entity_length, len(tokens))):
                start_pieces = token_to_wordpiece.get(start)
                end_pieces = token_to_wordpiece.get(end)

                # 检查wordpiece映射是否存在
                if not start_pieces or not end_pieces:
                    continue

                actual_start = start_pieces[0]
                actual_end = end_pieces[-1]

                # 验证wordpiece索引有效性
                if (actual_end < actual_start or 
                    actual_start >= hidden_states.size(1) or 
                    actual_end >= hidden_states.size(1)):
                    continue

                try:
                    # 提取span的hidden states
                    span_hidden = hidden_states[0, actual_start:actual_end+1]  # [span_len, hidden_size]
                    
                    if span_hidden.size(0) == 0:
                        continue
                        
                    span_embedding = torch.mean(span_hidden, dim=0)  # 平均池化

                    feature_dict = {
                        'span_embedding': span_embedding.cpu().numpy().tolist(),
                        'span_length': end - start + 1,
                        'wordpiece_length': actual_end - actual_start + 1
                    }
                    
                    # 如果有logits，添加相关特征
                    if logits is not None:
                        span_logits = logits[0, actual_start:actual_end+1]  # [span_len, num_labels]
                        span_logits_pooled = torch.mean(span_logits, dim=0)  # 平均池化
                        
                        # 计算softmax概率
                        span_softmax = torch.softmax(span_logits_pooled, dim=0)
                        
                        # 预测标签和置信度
                        pred_label_idx = torch.argmax(span_softmax)
                        pred_score = span_softmax[pred_label_idx].item()
                        
                        # 计算熵（不确定性）
                        span_entropy = -torch.sum(span_softmax * torch.log(span_softmax + 1e-8)).item()
                        
                        feature_dict.update({
                            'logits_pooled': span_logits_pooled.cpu().numpy().tolist(),
                            'softmax_pooled': span_softmax.cpu().numpy().tolist(), 
                            'pred_score': pred_score,
                            'span_entropy': span_entropy
                        })
                    
                    span_features[(start, end)] = feature_dict
                    
                except Exception as e:
                    print(f"计算span({start}, {end})特征时出错: {e}")
                    continue
        
        return span_features
        
    except Exception as e:
        print(f"提取span特征失败: {e}")
        import traceback
        print(traceback.format_exc())
        return {}
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
        
        # 提取实体级别特征（包含模型特征）
        entity_features = extract_entity_level_features(tokens, true_tags, pred_tags, 
                                                       model, tokenizer, device, max_length)
        
        # 计算序列级别统计
        token_accuracy = sum(1 for t, p in zip(true_tags, pred_tags) if t == p) / len(true_tags)
        
        # 计算实体统计
        pred_entities = [f for f in entity_features if f['match_type'] != 'missed']#预测的实体数
        true_entities_from_labels = extract_entities(true_tags)  # 从真实标签提取的实体
        num_true_entities = len(true_entities_from_labels)  # 真实的实体数
        num_pred_entities = len(pred_entities)  # 预测的实体数
        correct_entities = len([f for f in entity_features if f['match_type'] == 'correct'])#正确的实体数
        
        # 详细错误统计
        error_stats = {
            'correct': len([f for f in entity_features if f['match_type'] == 'correct']),
            'typeerror': len([f for f in entity_features if f['match_type'] == 'typeerror']),
            'boundaryerror': len([f for f in entity_features if f['match_type'] == 'boundaryerror']),
            'botherror': len([f for f in entity_features if f['match_type'] == 'botherror']),
            'hallucinationerror': len([f for f in entity_features if f['match_type'] == 'hallucinationerror']),
            'missed': len([f for f in entity_features if f['match_type'] == 'missed'])
        }
        
        sample_id = sample.get('id') or sample.get('ID') or sample.get('Id') or sample.get('iD')
        if sample_id is None:
            sample_id = f"sample_{len(judge_samples)}"

        # 构造judge样本
        judge_sample = {
            'id': sample_id,
            'tokens': tokens,
            'true_labels': true_tags,
            'pred_labels': pred_tags,
            'entity_features': entity_features,
            'sequence_accuracy': token_accuracy,
            'sequence_length': len(tokens),
            'num_entities_pred': num_pred_entities,
            'num_entities_true': num_true_entities,
            'num_correct_entities': correct_entities,
            'error_statistics': error_stats
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
        fold_true_entities = sum(sample['num_entities_true'] for sample in fold_judge_samples)
        fold_pred_entities = sum(sample['num_entities_pred'] for sample in fold_judge_samples)
        fold_correct_entities = sum(sample['num_correct_entities'] for sample in fold_judge_samples)
        fold_accuracy = fold_correct_entities / fold_true_entities if fold_true_entities > 0 else 0
        
        # 计算折级错误统计
        fold_error_stats = {
            'correct': sum(sample['error_statistics']['correct'] for sample in fold_judge_samples),
            'typeerror': sum(sample['error_statistics']['typeerror'] for sample in fold_judge_samples),
            'boundaryerror': sum(sample['error_statistics']['boundaryerror'] for sample in fold_judge_samples),
            'botherror': sum(sample['error_statistics']['botherror'] for sample in fold_judge_samples),
            'hallucinationerror': sum(sample['error_statistics']['hallucinationerror'] for sample in fold_judge_samples),
            'missed': sum(sample['error_statistics']['missed'] for sample in fold_judge_samples)
        }
        
        fold_stats = {
            "fold_id": fold_id + 1,
            "model_path": model_path,
            "train_samples": len(fold_train),
            "test_samples": len(fold_test),
            "judge_samples": len(fold_judge_samples),
            "true_entities": fold_true_entities,
            "pred_entities": fold_pred_entities,
            "correct_entities": fold_correct_entities,
            "entity_accuracy": fold_accuracy,
            "avg_sequence_accuracy": sum(sample['sequence_accuracy'] for sample in fold_judge_samples) / len(fold_judge_samples) if fold_judge_samples else 0,
            "error_breakdown": fold_error_stats
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
    total_true_entities = sum(sample['num_entities_true'] for sample in all_judge_samples)
    total_pred_entities = sum(sample['num_entities_pred'] for sample in all_judge_samples)
    correct_entities = sum(sample['num_correct_entities'] for sample in all_judge_samples)
    avg_sequence_acc = sum(sample['sequence_accuracy'] for sample in all_judge_samples) / len(all_judge_samples) if all_judge_samples else 0
    
    # 计算整体错误统计
    overall_error_stats = {
        'correct': sum(sample['error_statistics']['correct'] for sample in all_judge_samples),
        'typeerror': sum(sample['error_statistics']['typeerror'] for sample in all_judge_samples),
        'boundaryerror': sum(sample['error_statistics']['boundaryerror'] for sample in all_judge_samples),
        'botherror': sum(sample['error_statistics']['botherror'] for sample in all_judge_samples),
        'hallucinationerror': sum(sample['error_statistics']['hallucinationerror'] for sample in all_judge_samples),
        'missed': sum(sample['error_statistics']['missed'] for sample in all_judge_samples)
    }
    
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
            "total_true_entities": total_true_entities,
            "total_pred_entities": total_pred_entities,
            "correct_entities": correct_entities,
            "overall_entity_accuracy": correct_entities / total_true_entities if total_true_entities > 0 else 0,
            "overall_sequence_accuracy": avg_sequence_acc,
            "error_breakdown": overall_error_stats,
            "error_percentages": {
                "correct": overall_error_stats['correct'] / total_true_entities * 100 if total_true_entities > 0 else 0,
                "typeerror": overall_error_stats['typeerror'] / total_true_entities * 100 if total_true_entities > 0 else 0,
                "boundaryerror": overall_error_stats['boundaryerror'] / total_true_entities * 100 if total_true_entities > 0 else 0,
                "botherror": overall_error_stats['botherror'] / total_true_entities * 100 if total_true_entities > 0 else 0,
                "hallucinationerror": overall_error_stats['hallucinationerror'] / total_true_entities * 100 if total_true_entities > 0 else 0,
                "missed": overall_error_stats['missed'] / total_true_entities * 100 if total_true_entities > 0 else 0
            }
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
    print(f"真实实体数: {total_true_entities}")
    print(f"预测实体数: {total_pred_entities}")
    print(f"正确实体数: {correct_entities}")
    if total_true_entities > 0:
        print(f"实体准确率(基于真实实体): {correct_entities/total_true_entities:.3f}")
        if total_pred_entities > 0:
            print(f"实体精确率(基于预测实体): {correct_entities/total_pred_entities:.3f}")
        print(f"\n--- 详细错误分析 ---")
        print(f"完全正确(correct): {overall_error_stats['correct']} ({overall_error_stats['correct']/total_true_entities*100:.1f}%)")
        print(f"类型错误(typeerror): {overall_error_stats['typeerror']} ({overall_error_stats['typeerror']/total_true_entities*100:.1f}%)")
        print(f"边界错误(boundaryerror): {overall_error_stats['boundaryerror']} ({overall_error_stats['boundaryerror']/total_true_entities*100:.1f}%)")
        print(f"类型边界均错(botherror): {overall_error_stats['botherror']} ({overall_error_stats['botherror']/total_true_entities*100:.1f}%)")
        print(f"幻觉错误(hallucinationerror): {overall_error_stats['hallucinationerror']} ({overall_error_stats['hallucinationerror']/total_true_entities*100:.1f}%)")
        print(f"遗漏实体(missed): {overall_error_stats['missed']} ({overall_error_stats['missed']/total_true_entities*100:.1f}%)")
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