import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import config


class TextClassificationDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.Config["pretrain_model_path"])
        self.max_length = config.Config.get("max_seq_length", 128)
        self.labels_map = {"好评": 1, "差评": 0}  # 标签映射

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["review"]
        label = self.data.iloc[idx]["label"]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders():
    train_dataset = TextClassificationDataset(config.Config["train_data_path"])
    valid_dataset = TextClassificationDataset(config.Config["valid_data_path"])
    train_loader = DataLoader(train_dataset, batch_size=config.Config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.Config["batch_size"])
    return train_loader, valid_loader
