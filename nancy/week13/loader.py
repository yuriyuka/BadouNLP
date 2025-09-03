# -*- coding: utf-8 -*-

import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class NERDataset(Dataset):
    def __init__(self, data_path: str, config: Dict):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])
        self.label_list: List[str] = config["label_list"]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.max_length = config["max_length"]
        self.label_all_tokens = config.get("label_all_tokens", False)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if not line.strip():
                    continue
                eg = json.loads(line)
                tokens: List[str] = eg["tokens"]
                labels: List[str] = eg["labels"]

                assert len(tokens) == len(labels)

                enc = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                word_ids = enc.word_ids(batch_index=0)
                aligned_labels = self._align_labels(labels, word_ids)

                input_ids = enc["input_ids"][0]
                attention_mask = enc["attention_mask"][0]
                labels_tensor = torch.LongTensor(aligned_labels)

                self.data.append([input_ids, attention_mask, labels_tensor])

    def _align_labels(self, labels: List[str], word_ids: List[int]):
        aligned = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned.append(-100)
            elif word_idx != previous_word_idx:
                aligned.append(self.label2id[labels[word_idx]])
            else:
                if self.label_all_tokens:
                    label = labels[word_idx]
                    if label.startswith("B-"):
                        label = "I-" + label[2:]
                    aligned.append(self.label2id[label])
                else:
                    aligned.append(-100)
            previous_word_idx = word_idx
        # 截断或pad到max_length
        if len(aligned) < self.max_length:
            aligned += [-100] * (self.max_length - len(aligned))
        else:
            aligned = aligned[: self.max_length]
        return aligned

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(data_path: str, config: Dict, shuffle: bool = True):
    ds = NERDataset(data_path, config)
    dl = DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


