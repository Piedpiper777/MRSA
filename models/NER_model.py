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
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_hidden=False):
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
        
        if return_hidden:
            return sequence_output, lstm_output, logits
        
        if labels is not None:
            # 确保labels在正确设备上
            labels = labels.to(device)
            return self._compute_loss(logits, labels, attention_mask), logits
        
        return logits
    
    def _compute_loss(self, logits, labels, attention_mask):
        """简化版 CRF 损失：与可跑通脚本保持一致逻辑。
        规则：
          1) 使用 attention_mask 作为 CRF 的有效位置（包含 [CLS]/[SEP] 也无妨）
          2) 将 labels 中的 -100 替换为 0 (O) 以满足 CRF 不接受负数的限制
          3) 仅在需要时对齐长度
        """
        if attention_mask.shape != labels.shape:
            min_len = min(attention_mask.size(1), labels.size(1))
            attention_mask = attention_mask[:, :min_len]
            labels = labels[:, :min_len]
            logits = logits[:, :min_len, :]

        mask = attention_mask.bool()

        # 复制并替换 ignore_index=-100 为 0 (O 标签)
        crf_labels = labels.clone()
        # 检测非法（除 -100 外的越界）标签并回退为 O
        invalid_raw = (crf_labels != -100) & ((crf_labels < 0) | (crf_labels >= self.num_labels))
        if invalid_raw.any():
            # 打印一次警告（可按需去掉）
            print(f"警告: 发现越界标签 {crf_labels[invalid_raw].unique().tolist()}，已替换为 O")
            crf_labels[invalid_raw] = -100
        crf_labels[crf_labels == -100] = 0

        try:
            loss = -self.crf(logits, crf_labels, mask=mask, reduction='mean')
        except Exception as e:
            print(f"CRF计算失败，使用交叉熵损失: {e}")
            active = mask.view(-1)
            active_logits = logits.view(-1, self.num_labels)[active]
            active_labels = crf_labels.view(-1)[active]
            loss = F.cross_entropy(active_logits, active_labels)
        return loss
    
    def decode(self, logits, attention_mask=None):
        """改进的CRF解码"""
        if attention_mask is None:
            attention_mask = torch.ones(logits.shape[0], logits.shape[1], 
                                      dtype=torch.bool, device=logits.device)
        
        mask = attention_mask.bool()
        return self.crf.decode(logits, mask=mask)
    
    def predict(self, input_ids, attention_mask=None):
        """改进的预测函数"""
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