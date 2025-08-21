# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_ner.txt",
    "valid_data_path": "data/valid_ner.txt",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 128,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adamw",
    "learning_rate": 2e-5,
    "pretrain_model_path":r"E:\pretrain_models\bert-base-chinese",
    "seed": 987,
    # NER任务相关配置
    "num_labels": 9,  # BIO标注方案：O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC
    "label2id": {
        "O": 0,
        "B-PER": 1, "I-PER": 2,
        "B-LOC": 3, "I-LOC": 4,
        "B-ORG": 5, "I-ORG": 6,
        "B-MISC": 7, "I-MISC": 8
    },
    "id2label": {
        0: "O",
        1: "B-PER", 2: "I-PER",
        3: "B-LOC", 4: "I-LOC", 
        5: "B-ORG", 6: "I-ORG",
        7: "B-MISC", 8: "I-MISC"
    },
    # LoRA配置
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "key", "value"]
}