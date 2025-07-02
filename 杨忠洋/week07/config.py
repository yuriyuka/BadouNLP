# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./output/model.bin",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_seq_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 12,
    "epoch": 3,
    "batch_size": 50,
    "pooling_style": "average",
    "optimizer": "adam",
    "learning_rate":  5e-5,
    "pretrain_model_path": r"D:\AI\八斗精品班\第六周 语言模型和预训练\bert-base-chinese",
    "train_data_path": "./data/文本分类练习.csv",  # 修改为CSV路径
    "valid_data_path": "./data/文本分类练习.csv",  # 同上
    "num_labels": 2,  # 二分类任务
    "dropout_rate": 0.3,  # 添加dropout参数

}

