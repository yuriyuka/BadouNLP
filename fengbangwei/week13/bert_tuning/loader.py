# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    # 输入输出转化成序列
    def prepare_data(self, title, content):
        # 你好吗 sep 120
        input_max_length = self.config["input_max_length"]
        content_seq = self.tokenizer.encode(content, add_special_tokens=False)
        title_seq = self.tokenizer.encode(title, add_special_tokens=False)

        input_seq = [self.tokenizer.cls_token_id] + content_seq + [self.tokenizer.sep_token_id] + title_seq + [
            self.tokenizer.sep_token_id]
        gold = len(content_seq) * [-100] + [-100] + title_seq + [self.tokenizer.sep_token_id] + [-100]

        input_seq = input_seq[:input_max_length] + [0] * (input_max_length - len(input_seq))
        gold = gold[:input_max_length] + [0] * (input_max_length - len(gold))

        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(gold)])

        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dl = load_data(Config["train_data_path"], Config, 1)
    for batch in dl:
        print(batch)
        break
