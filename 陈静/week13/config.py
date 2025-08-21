# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_ner_sample.json",
    "valid_data_path": "data/valid_ner_sample.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "task_type": "ner",  
    "num_labels": 9, 
    "max_length": 128,  
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 2,
    "batch_size": 32,  
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 5e-5,  
    "pretrain_model_path": "bert-base-chinese",
    "seed": 987
}