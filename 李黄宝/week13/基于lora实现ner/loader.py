# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer

"""
数据加载 loader.py
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)
                
                input_id = torch.LongTensor(input_ids)
                label_index = torch.LongTensor(labels)
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text, padding=True):
        input_id = []
        for char in text:
                # 对单个字符进行tokenize
                tokens = self.tokenizer.tokenize(char)
                if tokens:  # 如果tokenize结果不为空
                    token_id = self.tokenizer.convert_tokens_to_ids(tokens[0])
                else:  # 如果没有对应的token，使用UNK
                    token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)[0]
                input_id.append(token_id)
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
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
    dg = DataGenerator(Config['train_data_path'], Config)
    print(dg[0])
