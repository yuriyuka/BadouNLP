import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.optim import Adam, SGD
from transformers import AutoModelForTokenClassification  # 修改模型类型

# 修改模型加载方式
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["pretrain_model_path"],
    num_labels=Config["class_num"],
    id2label=Config["id2label"],
    label2id=Config["label2id"]
)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
