# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    # "train_data_path": "data/train_tag_news.json",
    # "valid_data_path": "data/valid_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":"/Users/chenayu/Desktop/元芳/八斗学院/1/b1-其他/bert-base-chinese",
    "seed": 987,

    # "modelType":"ner", # tc or ner
    "schema_path": "/Users/chenayu/Desktop/元芳/八斗学院/1/a13-第十三周 大模型第三讲/wee13_practice/理解lora/data/schema.json",
    "train_data_path": "/Users/chenayu/Desktop/元芳/八斗学院/1/a13-第十三周 大模型第三讲/wee13_practice/理解lora/data/train",
    "valid_data_path": "/Users/chenayu/Desktop/元芳/八斗学院/1/a13-第十三周 大模型第三讲/wee13_practice/理解lora/data/test",
}