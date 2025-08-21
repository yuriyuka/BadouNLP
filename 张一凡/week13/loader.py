# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
NER数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.label_to_index = config["label_to_id"]
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.config["num_labels"] = len(self.label_to_index)
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
                text = line["text"]
                labels = line["labels"]

                # Tokenize text and align labels
                tokenized = self.tokenizer(text, truncation=True, max_length=self.config["max_length"],
                                           padding="max_length")
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Convert labels to token level labels
                token_labels = self.align_labels(text, labels, input_ids)

                input_ids = torch.LongTensor(input_ids)
                attention_mask = torch.LongTensor(attention_mask)
                labels = torch.LongTensor(token_labels)

                self.data.append([input_ids, attention_mask, labels])
        return

    def align_labels(self, text, original_labels, input_ids):
        # Convert original character-level labels to token-level labels
        token_labels = []
        char_to_token_map = self.tokenizer(text, return_offsets_mapping=True).char_to_token

        # Initialize all labels as 'O'
        token_labels = [self.label_to_index["O"]] * len(input_ids)

        # Special tokens ([CLS], [SEP], [PAD]) get label -100 (ignored by loss)
        for i, (input_id, offset) in enumerate(
                zip(input_ids, self.tokenizer(text, return_offsets_mapping=True)["offset_mapping"])):
            if input_id == self.tokenizer.pad_token_id:
                token_labels[i] = -100
            elif offset[0] == 0 and offset[1] == 0:  # Special tokens
                token_labels[i] = -100

        # Process each entity in original labels
        for label in original_labels:
            start, end, entity_type = label
            # Find all tokens that overlap with this entity
            for char_pos in range(start, end):
                token_pos = char_to_token_map(char_pos)
                if token_pos is not None:
                    # First token of entity gets B- label, others get I- label
                    if char_pos == start:
                        label_name = f"B-{entity_type}"
                    else:
                        label_name = f"I-{entity_type}"
                    token_labels[token_pos] = self.label_to_index.get(label_name, self.label_to_index["O"])

        return token_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
