# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载预训练BERT+分类层
        self.bert = BertForTokenClassification.from_pretrained(
            config["bert_path"],
            num_labels=config["class_num"],
            return_dict=True
        )

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
        self.bert = get_peft_model(self.bert, peft_config)
        # 确保分类层可训练
        for param in self.bert.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,
                         labels=labels)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
