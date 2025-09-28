"""
BERT-BiLSTM-CRF模型 - 只包含模型架构定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchcrf import CRF 

# MSRA数据集的标签映射
LABEL_MAP = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-LOC': 3, 'I-LOC': 4, 
    'B-ORG': 5, 'I-ORG': 6
}

ID2LABEL = {v: k for k, v in LABEL_MAP.items()}  # 反向映射
NUM_LABELS = len(LABEL_MAP)  # 标签数量

class BertBiLSTMCRF(nn.Module):
    """BERT-BiLSTM-CRF模型用于中文命名实体识别"""
    
    def __init__(self, num_labels=NUM_LABELS, bert_model_name='bert-base-chinese', 
                 lstm_hidden_size=256, lstm_layers=2, dropout_rate=0.1, max_length=128):
        super().__init__()
        
        # 模型配置
        self.num_labels = num_labels
        self.max_length = max_length
        self.lstm_hidden_size = lstm_hidden_size
        
        # 保存配置
        self.config = {
            'num_labels': num_labels,
            'bert_model_name': bert_model_name,
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_layers': lstm_layers,
            'dropout_rate': dropout_rate,
            'max_length': max_length
        }
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        # CRF层
        self.crf = CRF(num_labels, batch_first=True)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类层权重"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        # 兼容输入格式
        if attention_mask is None and isinstance(input_ids, (list, tuple)):
            input_ids, attention_mask = input_ids
        
        # 确保设备一致
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state
        
        # BiLSTM编码
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        # 分类层
        logits = self.classifier(lstm_output)
        
        if labels is not None:
            # 确保labels在正确设备上
            labels = labels.to(device)
            return self._compute_loss(logits, labels, attention_mask), logits
        
        return logits
    
    def _compute_loss(self, logits, labels, attention_mask):
        """改进版 CRF 损失计算

        改进要点：
          1) 仅把真正有标签的 token (labels != -100 且 attention_mask=1) 作为 CRF 有效位置，避免 [CLS]/[SEP] 参与状态转移学习。
          2) 对齐维度，防御性截断。
          3) 将 ignore_index(-100) / 越界标签统一映射为 0(O) 供 CRF 计算，但不会在 mask 中启用这些位置。
          4) 保证每个 batch 序列至少有一个 True 的 mask（CRF 要求第一步全部有效，如果第一步无效则显式强制其有效且标为 O）。
          5) 出错自动回退交叉熵，只在有效 token 上计算。
        """
        # ---- 维度对齐 ----
        if attention_mask.shape != labels.shape or logits.size(1) != labels.size(1):
            min_len = min(attention_mask.size(1), labels.size(1), logits.size(1))
            attention_mask = attention_mask[:, :min_len]
            labels = labels[:, :min_len]
            logits = logits[:, :min_len, :]

        # 基础 mask（去除 padding）
        base_mask = attention_mask.bool()
        # 有效标签位置（排除 ignore_index）
        label_valid = labels != -100
        # 仅真正需要训练的 token mask
        valid_mask = base_mask & label_valid

        # 处理越界标签：标记为 ignore
        invalid_mask = (labels != -100) & ((labels < 0) | (labels >= self.num_labels))
        if invalid_mask.any():
            print(f"警告: 发现 {invalid_mask.sum().item()} 个越界标签，已忽略")
            labels = labels.clone()
            labels[invalid_mask] = -100
            # 更新 valid_mask
            label_valid = labels != -100
            valid_mask = base_mask & label_valid

        # ---- 确保每个序列至少有一个 True ----
        # 若某些序列全部 False（极少见），强制第一个位置为 True，并把该标签视作 O
        row_has_token = valid_mask.any(dim=1)
        if (~row_has_token).any():
            need_fix = (~row_has_token).nonzero(as_tuple=False).flatten()
            if len(need_fix) > 0:
                valid_mask[need_fix, 0] = True
                # 更安全的标签修正
                if (labels[need_fix, 0] == -100).any():
                    labels = labels.clone()  # 确保不in-place修改
                    labels[need_fix, 0] = 0

        # CRF 接收的 mask
        crf_mask = valid_mask

        # 构造 CRF 标签：复制并把所有 ignore_index 换成 0
        crf_labels = labels.clone()
        crf_labels[crf_labels == -100] = 0

        # 保证第一列都是 True（CRF 内部实现常假设第一时刻有效）
        if not torch.all(crf_mask[:, 0]):
            # 只把第一列补 True，不改变其它列
            crf_mask[:, 0] = True
            # 对第一列被补上的（原本 False）标签设为 O
            need_o = (labels[:, 0] == -100)
            if need_o.any():
                crf_labels[need_o, 0] = 0

        try:
            loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
        except Exception as e:
            print(f"CRF计算失败，使用交叉熵损失回退: {e}")
            active = crf_mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)[active]
            active_labels = crf_labels.view(-1)[active]
            loss = F.cross_entropy(active_logits, active_labels)
        return loss
    
    def decode(self, logits, attention_mask=None):
        """CRF解码"""
        if attention_mask is None:
            attention_mask = torch.ones(logits.shape[0], logits.shape[1], 
                                      dtype=torch.bool, device=logits.device)
        
        mask = attention_mask.bool()
        return self.crf.decode(logits, mask=mask)
    
    def predict(self, input_ids, attention_mask=None):
        """预测函数"""
        self.eval()
        
        # 设备处理
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = self.decode(logits, attention_mask)
            probs = F.softmax(logits, dim=-1)
        
        return predictions, probs
    
    # 添加一个有用的方法来获取模型信息
    def get_model_info(self):
        """获取模型详细信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'BERT-BiLSTM-CRF',
            'num_labels': self.num_labels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'bert_hidden_size': self.bert.config.hidden_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'lstm_layers': self.lstm.num_layers,
            'is_bidirectional': self.lstm.bidirectional,
            'dropout_rate': self.dropout.p,
            'max_length': self.max_length
        }
    
    def freeze_bert_layers(self, freeze_layers=None):
        """冻结BERT的部分层
        
        Args:
            freeze_layers: 要冻结的层数，如果为None则冻结所有BERT参数
        """
        if freeze_layers is None:
            # 冻结所有BERT参数
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # 冻结指定层数
            # 冻结嵌入层
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            
            # 冻结前freeze_layers层编码器
            for i in range(min(freeze_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False
        
        return self