# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import json
import re
from transformers import BertTokenizer, BertModel, BertConfig


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, add_pooling_layer=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, attention_mask=None, y=None):
        if y is not None:
            outputs = self.bert(x, attention_mask=attention_mask)
            sequence_output = outputs[0]
            y_pred = self.classify(sequence_output)

            # Reshape for loss calculation
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            outputs = self.bert(x, attention_mask=attention_mask)
            sequence_output = outputs[0]
            y_pred = self.classify(sequence_output)
            return torch.softmax(y_pred, dim=-1)


def load_news_data(json_path):
    """加载新闻标题数据，返回标题列表"""
    titles = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            title = data.get("title", "").strip()
            if title:  # 确保标题不为空
                titles.append(title)
    return titles


def build_sample(tokenizer, window_size, titles):
    """从标题列表中随机构建一个样本"""
    title = random.choice(titles)
    title_ids = tokenizer.encode(title, add_special_tokens=False)

    # 如果标题太短，填充或跳过
    if len(title_ids) < window_size + 1:
        # 填充到window_size+1长度
        padding = [tokenizer.pad_token_id] * (window_size + 1 - len(title_ids))
        title_ids = title_ids + padding

    start_idx = random.randint(0, max(0, len(title_ids) - window_size - 1))
    input_ids = title_ids[start_idx:start_idx + window_size]
    target_ids = title_ids[start_idx + 1:start_idx + window_size + 1]

    # 创建注意力掩码（1表示实际token，0表示填充）
    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in input_ids]

    return input_ids, target_ids, attention_mask


def build_dataset(sample_length, tokenizer, window_size, titles):
    """建立训练数据集"""
    dataset_x = []
    dataset_y = []
    attention_masks = []

    for i in range(sample_length):
        x, y, mask = build_sample(tokenizer, window_size, titles)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_masks.append(mask)

    return (
        torch.LongTensor(dataset_x),
        torch.LongTensor(dataset_y),
        torch.FloatTensor(attention_masks)
    )


def build_model(vocab_size, hidden_size, pretrain_model_path):
    """初始化语言模型"""
    model = LanguageModel(hidden_size, vocab_size, pretrain_model_path)
    return model


def generate_sentence(openings, model, tokenizer, window_size):
    """生成文本"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # 初始化输入
        input_ids = tokenizer.encode(openings, add_special_tokens=False)
        if len(input_ids) > window_size:
            input_ids = input_ids[-window_size:]

        attention_mask = [1] * len(input_ids)

        # 填充输入
        if len(input_ids) < window_size:
            padding = [tokenizer.pad_token_id] * (window_size - len(input_ids))
            input_ids = input_ids + padding
            attention_mask = attention_mask + [0] * (window_size - len(attention_mask))

        generated_ids = list(input_ids)

        # 生成文本
        for _ in range(50):  # 最多生成50个token
            inputs = torch.LongTensor([input_ids]).to(device)
            attn_mask = torch.FloatTensor([attention_mask]).to(device)

            probs = model(inputs, attention_mask=attn_mask)
            last_token_probs = probs[0, -1, :]

            # 采样下一个token
            next_token_id = sampling_strategy(last_token_probs.cpu().numpy())

            # 更新输入
            generated_ids.append(next_token_id)
            input_ids = input_ids[1:] + [next_token_id]
            attention_mask = attention_mask[1:] + [1]

        # 解码生成的文本
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text


def sampling_strategy(prob_distribution):
    """采样策略"""
    if random.random() > 0.1:
        return int(torch.argmax(torch.tensor(prob_distribution)))
    else:
        probs = np.exp(prob_distribution - np.max(prob_distribution))
        probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)


def train(json_path, save_weight=True):
    """训练函数"""
    epoch_num = 20
    batch_size = 32
    train_sample = 10000
    window_size = 20
    learning_rate = 5e-5
    pretrain_model_path = "bert-base-chinese"

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    vocab_size = tokenizer.vocab_size

    # 加载新闻标题数据
    titles = load_news_data(json_path)
    print(f"加载 {len(titles)} 条新闻标题")

    model = build_model(vocab_size, 768, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        steps = int(np.ceil(train_sample / batch_size))

        for step in range(steps):
            # 构建批量数据
            x, y, attn_masks = build_dataset(
                batch_size, tokenizer, window_size, titles
            )

            if torch.cuda.is_available():
                x, y, attn_masks = x.cuda(), y.cuda(), attn_masks.cuda()

            optimizer.zero_grad()
            loss = model(x, attention_mask=attn_masks, y=y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch + 1} | Step {step + 1}/{steps} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / steps
        print(f"Epoch {epoch + 1}/{epoch_num} | Avg Loss: {avg_loss:.4f}")

        # 生成示例
        print("生成示例:")
        print(generate_sentence("阿根廷", model, tokenizer, window_size))
        print(generate_sentence("国际通用航空", model, tokenizer, window_size))
        print(generate_sentence("北京实施", model, tokenizer, window_size))

    if save_weight:
        torch.save(model.state_dict(), "news_title_model.pth")


if __name__ == "__main__":
    # 使用新闻数据训练
    news_data_path = r"D:\worksapce\ai_workspace\nlp20\week10\week10 文本生成问题\transformers-生成文章标题\sample_data.json"
    train(news_data_path)
