# -*- coding: utf-8 -*-
import torch
from config import Config
from transformers import BertTokenizer

from deepseek.week13.bert_tuning.main import build_model
from deepseek.week13.bert_tuning.common import generate_sentence

"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.model = build_model(len(self.tokenizer.vocab), config)
        state_dict = self.model.state_dict()
        loaded_weight = self.loaded_tuning_tactics_weights(config)
        state_dict.update(loaded_weight)
        self.model.load_state_dict(state_dict)
        print("模型加载完毕!")

    def predict(self, sentence):
        labeled_sentence = generate_sentence(sentence, self.model, self.tokenizer)
        # 去除多余空格并清理输出
        input_text = labeled_sentence.split("[SEP]")[0].strip()
        output_text = labeled_sentence.split("[SEP]")[1].strip()
        # 去除中文字符间的空格
        input_text = "".join(input_text.split())
        output_text = "".join(output_text.split())
        # print(input_text)
        # print(output_text)
        return input_text.replace('[CLS]', '') + ' -> ' + output_text

    def loaded_tuning_tactics_weights(self, config):
        tuning_tactics = config["tuning_tactics"]
        # 将微调部分权重加载
        if tuning_tactics == "lora_tuning":
            loaded_weight = torch.load('output/lora_tuning.pth')
        elif tuning_tactics == "p_tuning":
            loaded_weight = torch.load('output/p_tuning.pth')
        elif tuning_tactics == "prompt_tuning":
            loaded_weight = torch.load('output/prompt_tuning.pth')
        elif tuning_tactics == "prefix_tuning":
            loaded_weight = torch.load('output/prefix_tuning.pth')
        return loaded_weight


if __name__ == "__main__":
    sl = SentenceLabel(Config)

    sentence = "地球公转周期是多久"
    res = sl.predict(sentence)
    print(res)

    sentence = "水由什么元素组成"
    res = sl.predict(sentence)
    print(res)

    sentence = "光速是多少"
    res = sl.predict(sentence)
    print(res)

    sentence = "AI是什么意思"
    res = sl.predict(sentence)
    print(res)

    sentence = "空气中氧气含量多少"
    res = sl.predict(sentence)
    print(res)