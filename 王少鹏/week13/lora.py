# -*- coding: utf-8 -*-
"""
基于lora的ner任务训练
"""
import json
import os
import random
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from peft import get_peft_model, LoraConfig, TaskType
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


# ===================== 配置类 =====================
class Config:
    """训练与模型配置"""
    # 数据路径
    model_path = "output"  # 模型保存目录
    train_data_path = "data/train_ner.json"  # 训练集路径
    valid_data_path = "data/valid_ner.json"  # 验证集路径

    # 模型相关
    pretrain_model_path = "bert-base-chinese"  # 预训练模型路径/名称
    model_type = "bert"  # 模型类型（可扩展）
    num_labels = 9  # 标签数量
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                  "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

    # 训练参数
    max_length = 128  # 最大序列长度
    epoch = 10  # 训练轮数
    batch_size = 32  # 批大小
    learning_rate = 2e-5  # 学习率
    optimizer = "adamw"  # 优化器
    tuning_tactics = "lora_tuning"  # 微调策略: full_finetune / lora_tuning
    seed = 42  # 随机种子


# ===================== 随机种子设置 =====================
def set_seed(seed: int):
    """固定随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.seed)


# ===================== 数据集定义 =====================
class NERDataset(Dataset):
    """NER 数据集类"""
    def __init__(self, data_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        # 标签到ID映射
        self.label_map = {label: i for i, label in enumerate(Config.label_list)}
        self.pad_label_id = self.label_map["O"]  # 用 O 作为padding标签

    def load_data(self, path):
        """加载数据文件（JSON格式）"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回单条数据的编码结果"""
        item = self.data[idx]
        tokens = item["tokens"]  # 字/词列表
        labels = item["labels"]  # 标签列表

        # 使用tokenizer编码
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,  # 告诉tokenizer输入是词列表
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 对齐标签到token级别
        word_ids = encoding.word_ids()
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 特殊token忽略
            elif word_idx != previous_word_idx:
                label_ids.append(self.label_map[labels[word_idx]])
            else:
                # 同一个词的后续token使用I-标签（如果是BIO标注体系）
                label_ids.append(self.label_map[labels[word_idx]] if labels[word_idx].startswith("I")
                                 else self.label_map["I-" + labels[word_idx][2:]])
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids)
        }


# ===================== 模型定义 =====================
class NERModel(torch.nn.Module):
    """BERT+TokenClassification 模型（支持LoRA微调）"""
    def __init__(self):
        super().__init__()
        # 加载预训练BERT+分类层
        self.bert = BertForTokenClassification.from_pretrained(
            Config.pretrain_model_path,
            num_labels=Config.num_labels,
            return_dict=True
        )

        # 如果选择LoRA微调
        if Config.tuning_tactics == "lora_tuning":
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]
            )
            self.bert = get_peft_model(self.bert, peft_config)
            # 确保分类层可训练
            for param in self.bert.classifier.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         labels=labels)


# ===================== 评估器 =====================
class Evaluator:
    """验证集评估工具"""
    def __init__(self, model, tokenizer, valid_loader, device):
        self.model = model
        self.tokenizer = tokenizer
        self.valid_loader = valid_loader
        self.device = device
        self.logger = logging.getLogger(__name__)

    def evaluate(self):
        """计算验证集指标"""
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in self.valid_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                # 只取非padding位置
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, Config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]

                preds = torch.argmax(active_logits, dim=1).cpu().numpy()
                labels_ids = active_labels.cpu().numpy()

                pred_labels = [Config.label_list[p] for p in preds]
                true_labels_batch = [Config.label_list[l] for l in labels_ids]

                predictions.extend(pred_labels)
                true_labels.extend(true_labels_batch)

        # 计算指标
        precision = precision_score([true_labels], [predictions])
        recall = recall_score([true_labels], [predictions])
        f1 = f1_score([true_labels], [predictions])

        self.logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        self.logger.info("\n" + classification_report([true_labels], [predictions]))
        return f1


# ===================== 训练函数 =====================
def train_model(model, train_loader, valid_loader, tokenizer):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    evaluator = Evaluator(model, tokenizer, valid_loader, device)

    best_f1 = 0.0
    for epoch in range(Config.epoch):
        model.train()
        total_loss = 0
        logger.info(f"Epoch {epoch + 1}/{Config.epoch}")

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        f1 = evaluator.evaluate()
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(os.path.join(Config.model_path, "best_model"))
            tokenizer.save_pretrained(os.path.join(Config.model_path, "best_model"))
            logger.info("Saved best model")

    logger.info(f"Training complete. Best F1: {best_f1:.4f}")


# ===================== 主程序入口 =====================
def main():
    tokenizer = BertTokenizer.from_pretrained(Config.pretrain_model_path)
    train_dataset = NERDataset(Config.train_data_path, tokenizer, Config.max_length)
    valid_dataset = NERDataset(Config.valid_data_path, tokenizer, Config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)

    model = NERModel()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage: {trainable_params / total_params * 100:.2f}%")

    train_model(model, train_loader, valid_loader, tokenizer)


if __name__ == "__main__":
    main()
