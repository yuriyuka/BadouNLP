# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_ner.json",
    "valid_data_path": "data/valid_ner.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 128,
    "hidden_size": 256,
    "num_labels": 9,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 32,
    "tuning_tactics":"lora_tuning",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":r"E:\pretrain_models\bert-base-chinese",
    "seed": 987,
    "label_to_id": {  # NER标签映射
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-TIME": 7,
        "I-TIME": 8
    }
}
