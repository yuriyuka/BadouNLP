# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.json",
    "valid_data_path": "ner_data/test.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 128,
    "hidden_size": 512,
    "kernel_size": 3,
    "num_layers": 9,
    "epoch": 10,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"./bert-base-chinese",
    "seed": 987,
    "use_crf": True,
    "class_num": 9
}