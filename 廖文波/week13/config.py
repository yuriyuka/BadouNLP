# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "week13 大语言模型相关第三讲/work/model_output",
    "schema_path": "week13 大语言模型相关第三讲/work/ner_data/schema.json",
    "train_data_path": "week13 大语言模型相关第三讲/work/ner_data/train",
    "valid_data_path": "week13 大语言模型相关第三讲/work/ner_data/test",
    "vocab_path": "week13 大语言模型相关第三讲/work/chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "tuning_tactics": "lora_tuning",  # 可选值: lora_tuning, p_tuning, prompt_tuning, prefix_tuning
    "bert_path": r"/home/nbs07/model/bert-base-chinese"
}

