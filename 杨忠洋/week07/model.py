import torch
import torch.nn as nn
from transformers import BertModel

import config


class BERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.Config["pretrain_model_path"])
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # 扩展attention_mask维度用于广播
        expanded_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # 将padding位置的输出置为0
        masked_output = sequence_output * expanded_mask

        # 计算每个序列的实际长度（忽略padding）
        seq_lengths = expanded_mask.sum(dim=1)  # [batch_size, 1]

        # 避免除零错误（如果序列全为padding，则长度设为1）
        seq_lengths = torch.clamp(seq_lengths, min=1e-9)

        # 对非padding位置的向量求平均
        pooled_output = masked_output.sum(dim=1) / seq_lengths  # [batch_size, hidden_size]

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
