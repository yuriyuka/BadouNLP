# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        bert_path = config['bert_path']
        # bert
        self.bert = BertModel.from_pretrained(bert_path)
        # lstm
        self.layer = nn.LSTM(768, 256, bidirectional=True, num_layers=1)  # 单层LSTM
        # 分类
        self.classify = nn.Linear(hidden_size * 2, class_num)

        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.norm = nn.LayerNorm(hidden_size * 2)  # 双向LSTM输出维度是 hidden_size*2
        # loss采用交叉熵损失
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        bert_output = self.bert(x)
        x = bert_output.last_hidden_state  # (batch, seq_len, 768)


        lstm_out, _ = self.layer(x)
        x = x[:, :, :lstm_out.size(-1)]  # 对齐维度（BERT 768 → LSTM 512）
        x = self.norm(lstm_out + x)  # 保留原始特征


        predict = self.classify(x)  # (batch, seq_len, class_num)


        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                decoded = self.crf_layer.decode(predict)
                return decoded
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
