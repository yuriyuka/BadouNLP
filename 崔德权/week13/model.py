import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, BertForTokenClassification, AutoConfig
from torch.optim import Adam, SGD, AdamW

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        model_config = AutoConfig.from_pretrained(config["pretrain_model_path"], num_labels=config["num_labels"])
        self.bert = BertForTokenClassification.from_pretrained(
            config["pretrain_model_path"],
            config=model_config
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        return AdamW(model.parameters(), lr=learning_rate)
