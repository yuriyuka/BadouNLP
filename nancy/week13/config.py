# -*- coding: utf-8 -*-

"""
配置参数信息（NER任务，LoRA微调）
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_ner.json",
    "valid_data_path": "data/valid_ner.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "pretrain_model_path": "bert-base-chinese",
    "max_length": 128,
    "epoch": 5,
    "batch_size": 16,
    "learning_rate": 5e-5,
    "optimizer": "adam",
    "seed": 42,
    "tuning_tactics": "lora_tuning",
    "label_list": [
        "O",
        "B-PER", "I-PER",
        "B-ORG", "I-ORG",
        "B-LOC", "I-LOC"
    ],
    "label_all_tokens": False,
}

