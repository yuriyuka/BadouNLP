# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
# 从库里引入Tokenizer
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config["pretrain_model_path"])
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
                sentence = "".join(sentenece)
                self.sentences.append(sentence)
                input_ids = self.encode_sentence(sentenece)
                labels = self.padding(labels, -1)

                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        # 如果采用tokenizer的encoder方法的话，不会把话切成一个字符一个字符
        # 数字可能是一段一段的
        # 训练数据做序列标注，训练数据里面，每一个字符都是分开的
        # 不会把话切成一个字符一个字符，尤其是对于数字和英文
        # 两个字符对应一个label，处理要做对应的处理
        # 数字可能好几个数字一个字符，英文也是，好几个字母一个字符
        # 训练数据中，每一个数字都占一个字符
        # 两个字符对应一个label
        # 有可能出现bert词表以外的字符，这时按照UNK处理
        # 使用tokenizer的词表，每个字符都取出来，但是会有词表以外的字符，这个时候看做[UNK]
        input_ids = [self.tokenizer.vocab.get(char, self.tokenizer.vocab["[UNK]"]) for char in text]
        if padding:
            input_ids = self.padding(input_ids, self.tokenizer.vocab["[PAD]"])
        return input_ids

    # 补齐或截断输入的序列，使其可以在一个batch内运算
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


def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size=32)
    for x, y in dl:
        print(x.shape, y.shape)
        print(x[1], y[1])
        input()
