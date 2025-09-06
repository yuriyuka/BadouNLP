# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",  # 新目录存储NER模型
    "train_data_path": "data/train",  # NER训练数据（token级）
    "valid_data_path": "data/test",   # NER测试数据
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 128,  # NER句子通常更长，适当增大
    "hidden_size": 768,  # BERT-base隐藏层维度
    "epoch": 20,  # NER需要更多轮次训练
    "batch_size": 32,
    "tuning_tactics": "lora_tuning",
    "learning_rate": 2e-4,  # LoRA学习率可略高
    "pretrain_model_path": "bert-base-chinese",
    "seed": 987,
    "num_labels": 9,  # schema.json中标签总数
    "optimizer": "adam",  # 添加这一行，可选值："adam" 或 "sgd"
    "learning_rate": 2e-4,
}