# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BertModel

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        if config["model_type"] == "bert":
            self.layer = BertModel.from_pretrained(config["bert_path"], return_dict=False)
            hidden_size = self.layer.config.hidden_size
            self.classify = nn.Linear(hidden_size, class_num)
            self.use_bert = True
            lora_config = LoraConfig(
                r=8,  # 低秩矩阵秩
                lora_alpha=32,  # 缩放因子
                target_modules=["query", "value", "key"],  # LoRA 作用的层
                task_type=TaskType.SEQ_CLS  # 序列标注任务
            )
            self.bert = get_peft_model(self.bert, lora_config)
            self.classify = nn.Linear(hidden_size, class_num)
            self.use_bert = True
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.embedding(x)  #input shape:(batch_size, sen_len)
        x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
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
