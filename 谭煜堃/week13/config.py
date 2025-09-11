# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train",  # Updated for NER
    "valid_data_path": "data/valid",  # Updated for NER
    "schema_path": "data/schema.json",    # Path to label schema
    "model_type":"bert",
    "max_length": 128,
    "hidden_size": 768,
    "num_layers": 12,
    "epoch": 10,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "pretrain_model_path": "bert-base-chinese",
    "seed": 987,
    "num_labels": 9,  # Correct number of NER tags from schema.json
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
}