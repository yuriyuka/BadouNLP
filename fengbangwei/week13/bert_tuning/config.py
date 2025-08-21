# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "input_max_length": 120,
    "output_max_length": 30,
    # "input_max_length": 50,
    # "output_max_length": 30,
    "epoch": 50,
    "batch_size": 32,
    "valid_batch_size": 1,
    "tuning_tactics": "lora_tuning",  # 调优策略
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 42,
    "vocab_size": 6219,
    "vocab_path": "vocab.txt",
    "train_data_path": r"data\sample_data.json",
    "valid_data_path": r"data\valid_data.json",
    "beam_size": 5,
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\AI\nlp\bert-base-chinese"
}
