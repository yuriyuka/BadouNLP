import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForTokenClassification
from torch.optim import Adam, SGD

# 根据任务类型选择不同的模型
if Config.get("task_type", "classification") == "ner":
    TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=Config.get("num_labels", 9))
else:
    TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
