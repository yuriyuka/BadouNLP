# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "F:\BaiduNetdiskDownload\八斗精品班\第十三周\week13 大语言模型相关第三讲\训练大模型\理解lora\data\train_tag_news.json",
    "valid_data_path": "F:\BaiduNetdiskDownload\八斗精品班\第十三周\week13 大语言模型相关第三讲\训练大模型\理解lora\data\valid_tag_news.json",
    "vocab_path":"F:\BaiduNetdiskDownload\八斗精品班\第十三周\week13 大语言模型相关第三讲\训练大模型\理解lora\chars.txt",
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
    "pretrain_model_path":r"F:\BaiduNetdiskDownload\八斗精品班\第六周 语言模型\bert-base-chinese",
    "seed": 987,
    "class_num": 9,  # 根据具体NER任务调整实体类型数量
    "label2id": {  # 定义实体标签映射
        "O": 0,
        "B-PER": 1, "I-PER": 2,
        "B-LOC": 3, "I-LOC": 4,
        "B-ORG": 5, "I-ORG": 6,
        "B-MISC": 7, "I-MISC": 8
    }
}
Config["id2label"] = {v:k for k,v in Config["label2id"].items()}