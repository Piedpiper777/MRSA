"""
使用训练好的 final_model 进行预测的脚本。

更新要点 (2025-09-28):
- 修正: tokenizer 不能从 final_model 目录加载 (该目录只有自定义 config.json 与权重), 必须从原始预训练 BERT 目录加载 (包含 vocab.txt / tokenizer.json)。
- 新增: 支持显式传入 bert_model_dir 覆盖 config.json 里的 bert_model_name。
- 新增: 更友好的缺失文件检查与错误提示。

目录含义:
- model_dir: 训练保存的 final_model 目录 (包含 pytorch_model.bin + 自定义 config.json)
- bert_model_dir: 预训练 BERT 目录 (需包含 vocab.txt 与 tokenizer.json 或其中之一)
"""
import os
import json
import argparse
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
from dataclasses import dataclass
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 复用模型与标签映射
from models.NER_model import BertBiLSTMCRF, LABEL_MAP, ID2LABEL, NUM_LABELS

IGNORE_INDEX = -100

# ---------------- 新增: 读取并校验自定义 config ----------------

def load_custom_config(model_dir: str) -> Dict[str, Any]:
    """加载训练时保存的自定义配置 (不含 HuggingFace 原始 BERT 配置) 并做基础校验。"""
    cfg_path = os.path.join(model_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"未找到自定义配置文件: {cfg_path}")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # 基础字段检查
    required = ["num_labels", "bert_model_name"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"配置缺失必要字段: {missing}, 请确认训练阶段保存的 config.json 完整性")
    return cfg


def load_model(model_dir: str, bert_model: str, device: torch.device) -> BertBiLSTMCRF:
    """加载训练好的模型 (依赖保存的 config.json + state_dict)。"""
    config_path = os.path.join(model_dir, "config.json")
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.isfile(config_path) or not os.path.isfile(state_path):
        raise FileNotFoundError(f"模型目录不完整: {model_dir}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # 允许用户用 bert_model 覆盖
    bert_model_name = bert_model if bert_model else cfg.get('bert_model_name')
    if not bert_model_name:
        raise ValueError("config.json 中未包含 bert_model_name 且未显式传入 bert_model")
    model = BertBiLSTMCRF(
        num_labels=cfg.get('num_labels', NUM_LABELS),
        bert_model_name=bert_model_name,
        lstm_hidden_size=cfg.get('lstm_hidden_size', 256),
        lstm_layers=cfg.get('lstm_layers', 2),
        dropout_rate=cfg.get('dropout_rate', 0.1),
        max_length=cfg.get('max_length', 128)
    )
    try:
        sd = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        # 兼容老版本 PyTorch 无 weights_only 参数的情况
        sd = torch.load(state_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


def tokenize_with_alignment(tokenizer, tokens: List[str], max_length: int):
    """与训练阶段一致的分词 & 对齐逻辑。输入已分好词的 tokens。"""
    encodings = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    word_ids = encodings.word_ids()
    return encodings, word_ids


def decode_bio(pred_ids: List[int]) -> List[str]:
    return [ID2LABEL.get(int(i), 'O') for i in pred_ids]


def extract_entities(tokens: List[str], bio_tags: List[str]) -> List[Dict[str, Any]]:
    """从 BIO 序列中提取实体列表。"""
    entities = []
    current = None
    for idx, (tok, tag) in enumerate(zip(tokens, bio_tags)):
        if tag.startswith('B-'):
            if current:
                entities.append(current)
            current = {'type': tag[2:], 'tokens': [tok], 'start': idx, 'end': idx}
        elif tag.startswith('I-') and current and current['type'] == tag[2:]:
            current['tokens'].append(tok)
            current['end'] = idx
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    # 合成文本
    for ent in entities:
        ent['text'] = ''.join(ent['tokens'])
    return entities


def predict_one(model: BertBiLSTMCRF, tokenizer, raw_tokens: List[str], device: torch.device, max_length: int):
    encodings, word_ids = tokenize_with_alignment(tokenizer, raw_tokens, max_length)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        decoded = model.decode(logits, attention_mask)[0]  # list[int]
    # 只还原到原始 token 级别: 取该词第一个子词对应的预测
    ori_tags: List[str] = []
    seen = set()
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid in seen:
            continue
        seen.add(wid)
        ori_tags.append(ID2LABEL.get(decoded[pos], 'O'))
    # 截断到真实 token 长度
    ori_tags = ori_tags[:len(raw_tokens)]
    entities = extract_entities(raw_tokens, ori_tags)
    return {
        'tokens': raw_tokens,
        'tags': ori_tags,
        'entities': entities
    }


def read_inputs(args) -> List[List[str]]:
    if args.text:
        # 简单按空格切 (若用户未分词，可进一步接入分词器，这里保持轻量)
        tokens = args.text.strip().split()
        return [tokens]
    if not args.input_file:
        raise ValueError("必须提供 --text 或 --input_file")
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(args.input_file)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    if args.is_json:
        data = json.loads(content)
        all_tokens = []
        for item in data:
            if 'tokens' in item:
                all_tokens.append(item['tokens'])
            elif 'text' in item:
                all_tokens.append(item['text'].split())
        return all_tokens
    else:
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return [ln.split() for ln in lines]

@dataclass
class PredictConfig:
    model_dir: str = r'/workspace/models/saved_models/final_model'
    # 可覆盖 config.json 中的 bert_model_name, 留空则沿用 config
    bert_model_dir: str = ''
    input_file: str = r''  # 可为空表示走 --text 单句模式
    output_file: str = r''
    text: str = ''  # 单句直接输入
    is_json: bool = False

def _resolve_bert_dir(cfg: Dict[str, Any], override: str) -> str:
    cand = override.strip() if override else cfg.get('bert_model_name', '').strip()
    if not cand:
        raise ValueError("未提供 bert_model 目录 (缺少 override 且 config.json 无 bert_model_name)")
    if os.path.isdir(cand):
        return cand
    # 尝试 basename 匹配: 例如只放了最后一级目录名
    base = os.path.basename(cand.rstrip('/\\'))
    parent = os.path.dirname(cand.rstrip('/\\')) or '.'
    if os.path.isdir(base):
        return base
    # 环境变量回退
    env_dir = os.environ.get('BERT_MODEL_DIR')
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    raise FileNotFoundError(f"无法找到 BERT 目录: {cand}\n请确认: 1) 目录已挂载 2) 传入路径正确 3) 若已移动, 手动设置 PredictConfig.bert_model_dir 或环境变量 BERT_MODEL_DIR")

def main():
    # ---------------- 配置参数 (按需修改) ----------------
    args = PredictConfig(
        model_dir=r'/workspace/models/saved_models/final_model',
        bert_model_dir=r'/workspace/models/google-bert/bert-base-chinese',  # 若留空则使用 config.json 中路径
        input_file=r'/workspace/test.json',
        output_file=r'/workspace/predictions.json',
        text='',  # 若要单句直接预测, 在此填写文本 (空格分词或原句, 会按空格拆)
        is_json=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 读取训练时保存的自定义配置 (不含 HuggingFace 原始 BERT 配置)
    cfg = load_custom_config(args.model_dir)

    # 解析实际 BERT 目录
    bert_dir = _resolve_bert_dir(cfg, args.bert_model_dir)
    print(f"使用 BERT 目录: {bert_dir}")

    # ----------- tokenizer 加载 (来自原始 BERT 目录, 而不是 final_model) -----------
    # 需要 fast tokenizer 才能使用 word_ids()
    required_files = [
        os.path.join(bert_dir, 'tokenizer.json'),  # fast 优先
        os.path.join(bert_dir, 'vocab.txt')
    ]
    if not any(os.path.isfile(p) for p in required_files):
        raise FileNotFoundError(
            "未在 BERT 目录中找到 tokenizer.json 或 vocab.txt, 无法构建 tokenizer.\n" \
            f"检查目录: {bert_dir}\n" \
            "如果你只复制了部分文件, 需要从原始预训练模型目录拷贝完整 tokenizer 相关文件。"
        )

    tokenizer = AutoTokenizer.from_pretrained(bert_dir, use_fast=True)
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError('需要 fast tokenizer (包含 tokenizer.json)')

    # ----------- 加载自定义模型结构与权重 -----------
    model = load_model(args.model_dir, bert_dir, device)
    max_len = cfg.get('max_length', 128)

    # ----------- 准备输入 -----------
    # 如果 text 非空 => 单句模式优先
    if args.text:
        # 兼容未分词句子: 这里只简单用空格切, 如需更好分词可接入 jieba / pkuseg
        single_tokens = args.text.strip().split()
        batches = [single_tokens]
    else:
        batches = read_inputs(args)

    results = []
    for tokens in batches:
        pred = predict_one(model, tokenizer, tokens, device, max_len)
        results.append(pred)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已写入: {args.output_file}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
