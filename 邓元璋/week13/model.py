import torch.nn as nn
from config import Config
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.optim import Adam, SGD

# 初始化NER模型（token级分类）
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["pretrain_model_path"],
    num_labels=Config["num_labels"]  # 对应schema中的9个标签
)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)