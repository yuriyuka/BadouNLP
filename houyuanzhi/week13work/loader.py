# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # 添加标签映射
        self.label2id = config["label2id"]
        self.id2label = config.get("id2label", {v:k for k,v in self.label2id.items()})
        
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                content = line["content"]  # 使用content字段进行NER标注
                # 示例标注逻辑（需根据实际标注数据调整）
                labels = self.annotate_entities(content)  # 需要实现标注逻辑
                if self.config["model_type"] == "bert":
                    # 对齐token化后的标签
                    tokenized = self.tokenizer(
                        content, 
                        max_length=self.config["max_length"], 
                        padding="max_length", 
                        truncation=True,
                        return_offsets_mapping=True
                    )
                    aligned_labels = self.align_labels(tokenized, labels)
                    input_id = tokenized["input_ids"]
                    attention_mask = tokenized["attention_mask"]
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def annotate_entities(self, text):
        # 示例标注函数（需要替换为实际标注逻辑）
        labels = ["O"] * len(text)
        # 示例：简单匹配"北京"作为地名
        for i in range(len(text)-1):
            if text[i:i+2] == "北京":
                labels[i] = "B-LOC"
                labels[i+1] = "I-LOC"
        return labels

    def align_labels(self, tokenized, labels):
        # 对齐tokenized后的标签
        offset_mapping = tokenized["offset_mapping"]
        aligned_labels = []
        for offsets in offset_mapping:
            if offsets[0] == 0 and offsets[1] == 0:
                aligned_labels.append(self.label2id["O"])
            else:
                aligned_labels.append(self.label2id[labels[offsets[0]]])
        return aligned_labels

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
