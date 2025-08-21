# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train",
    "valid_data_path": "data/test",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 1,
    "batch_size": 1600,
    "use_crf": True,
    "class_num": 9,
    "tuning_tactics": "lora_tuning",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": "bert-base-chinese",
    "seed": 987
}
