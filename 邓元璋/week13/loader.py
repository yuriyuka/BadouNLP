# -*- coding: utf-8 -*-
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

# 加载NER标签映射（从schema.json）
def load_label_map(schema_path="data/schema.json"):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    label_to_index = schema  # schema中已是{"B-LOCATION":0, ...}
    index_to_label = {v: k for k, v in label_to_index.items()}
    return label_to_index, index_to_label

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])
        self.label_to_index, self.index_to_label = load_label_map()
        self.config["num_labels"] = len(self.label_to_index)
        self.data = self.load_data()

    def load_data(self):
        """加载NER数据（格式：每行"token 标签"，空行分隔句子）"""
        data = []
        with open(self.path, "r", encoding="utf-8") as f:
            sentence = []  # 存储一句的(token, label)
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if sentence:
                        data.append(self.process_sentence(sentence))
                        sentence = []
                else:
                    token, label = line.split()
                    sentence.append((token, label))
            if sentence:  # 处理最后一句
                data.append(self.process_sentence(sentence))
        return data

    def process_sentence(self, sentence):
        """处理单句：分词、标签对齐（解决BERT分词的subword问题）"""
        tokens, labels = zip(*sentence)
        # 1. BERT分词（可能拆分token为subword）
        tokenized = self.tokenizer(
            list(tokens),
            is_split_into_words=True,  # 已按token拆分
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # 2. 标签对齐：将原始标签映射到subword
        word_ids = tokenized.word_ids(batch_index=0)  # 每个subword对应的原始token索引
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:  # [CLS]或[SEP]
                aligned_labels.append(-100)  # 损失计算时忽略
            else:
                # 原始标签对应到subword，仅第一个subword保留B-标签，其余继承I-标签
                original_label = labels[word_id]
                aligned_labels.append(self.label_to_index[original_label])
        # 截断或补齐至max_length
        aligned_labels = aligned_labels[:self.config["max_length"]]
        aligned_labels += [-100] * (self.config["max_length"] - len(aligned_labels))
        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(attention_mask),
            torch.LongTensor(aligned_labels)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, shuffle=True):
    dataset = DataGenerator(data_path, config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("data/test", Config)
    print(dg[0])  # 测试输出：(input_ids, attention_mask, labels)