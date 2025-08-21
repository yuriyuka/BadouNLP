# -*- coding: utf-8 -*-

from typing import Dict
from transformers import AutoConfig, AutoModelForTokenClassification
from config import Config
from torch.optim import Adam, SGD


# 加载基础Token分类模型
def build_base_model():
    cfg = AutoConfig.from_pretrained(
        Config["pretrain_model_path"],
        num_labels=Config["num_labels"],
        id2label={i: l for i, l in enumerate(Config["label_list"])},
        label2id={l: i for i, l in enumerate(Config["label_list"])},
    )
    model = AutoModelForTokenClassification.from_pretrained(
        Config["pretrain_model_path"],
        config=cfg,
    )
    return model


def choose_optimizer(config: Dict, model):
    optimizer = config["optimizer"]
    learning_rate = float(config["learning_rate"])
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


