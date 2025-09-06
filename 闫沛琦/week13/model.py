# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF

from transformers import BertModel, AutoModelForTokenClassification
from config import Config

"""
建立网络模型结构
"""

class ConfigWrapper(object):
    def __init__(self, config):
        self.config = config
    
    def to_dict(self):
        return self.config


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        self.use_bert = config["use_bert"]
        self.config = ConfigWrapper(config)

        if(self.use_bert):
            self.layer = BertModel.from_pretrained(config["bert_path"])
            hidden_size = self.layer.config.hidden_size
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=False, num_layers=num_layers)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, target=None):
        if(self.use_bert):
            outputs = self.layer(input_ids)
            input_ids = outputs.last_hidden_state
        else:
            input_ids = self.embedding(input_ids)  #input shape:(batch_size, sen_len)        
            input_ids, _ = self.layer(input_ids)      #input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(input_ids) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


# TorchModel = AutoModelForTokenClassification.from_pretrained(
#     Config["bert_path"],
#     num_labels=Config["class_num"],  # NER 标签数量
#     return_dict=True
# )



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