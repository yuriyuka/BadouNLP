from torch.optim import Adam, SGD
from transformers import AutoModelForTokenClassification

from config import Config

TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"])


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)