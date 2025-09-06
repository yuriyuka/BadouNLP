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
        
        # 根据任务类型设置不同的标签映射
        if config.get("task_type") == "ner":
            # NER标签映射 (BIO格式)
            self.index_to_label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 
                                 5: 'B-ORG', 6: 'I-ORG', 7: 'B-MISC', 8: 'I-MISC'}
            self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
            self.config["class_num"] = len(self.index_to_label)
        else:
            # 文本分类标签映射
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
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                
                if self.config.get("task_type") == "ner":
                    # NER数据格式: {"text": "张三在北京工作", "labels": ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O"]}
                    text = line["text"]
                    labels = line["labels"]
                    
                    if self.config["model_type"] == "bert":
                        # 使用BERT tokenizer进行编码
                        encoding = self.tokenizer(text, max_length=self.config["max_length"], 
                                                padding="max_length", truncation=True, return_tensors="pt")
                        input_ids = encoding["input_ids"].squeeze(0)
                        
                        # 处理标签，考虑BERT的子词切分
                        label_ids = self.align_labels_with_tokens(text, labels, input_ids)
                        label_ids = torch.LongTensor(label_ids)
                    else:
                        input_ids = self.encode_sentence(text)
                        input_ids = torch.LongTensor(input_ids)
                        label_ids = torch.LongTensor([self.label_to_index.get(label, 0) for label in labels[:self.config["max_length"]]])
                        # 填充标签到最大长度
                        if len(label_ids) < self.config["max_length"]:
                            padding_length = self.config["max_length"] - len(label_ids)
                            label_ids = torch.cat([label_ids, torch.zeros(padding_length, dtype=torch.long)])
                    
                    self.data.append([input_ids, label_ids])
                    
                else:
                    # 原有的文本分类逻辑
                    tag = line["tag"]
                    label = self.label_to_index[tag]
                    title = line["title"]
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], padding="max_length", truncation=True)
                    else:
                        input_id = self.encode_sentence(title)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([label])
                    self.data.append([input_id, label_index])
        return

    def align_labels_with_tokens(self, text, labels, input_ids):
        """将字符级标签对齐到BERT的token级标签"""
        # 获取token对应的字符位置
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        aligned_labels = []
        
        # 添加[CLS]标签
        aligned_labels.append(0)  # [CLS] -> O
        
        char_to_label = {}
        char_idx = 0
        for i, label in enumerate(labels):
            if char_idx < len(text):
                char_to_label[char_idx] = self.label_to_index.get(label, 0)
                char_idx += 1
        
        # 处理中间的token
        current_char_idx = 0
        for i, token in enumerate(tokens[1:-1], 1):  # 跳过[CLS]和[SEP]
            if token.startswith('##'):
                # 子词，使用前一个token的标签
                if aligned_labels:
                    aligned_labels.append(aligned_labels[-1])
                else:
                    aligned_labels.append(0)
            else:
                # 新词的开始
                if current_char_idx < len(text):
                    aligned_labels.append(char_to_label.get(current_char_idx, 0))
                    current_char_idx += len(token.replace('##', ''))
                else:
                    aligned_labels.append(0)
        
        # 添加[SEP]标签
        aligned_labels.append(0)  # [SEP] -> O
        
        # 填充到最大长度
        while len(aligned_labels) < self.config["max_length"]:
            aligned_labels.append(0)  # padding -> O
            
        return aligned_labels[:self.config["max_length"]]

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
