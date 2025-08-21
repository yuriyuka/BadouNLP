# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train",
    "valid_data_path": "data/test",
    "vocab_path": r"F:\八斗ai课程\bert-base-chinese\vocab.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 100,
    "batch_size": 16,
    "tuning_tactics": "lora_tuning",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "class_num": 9,
    "pretrain_model_path": r"F:\八斗ai课程\bert-base-chinese",
    "seed": 987   # 随机数种子  用于复现
}

