# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel, BertTokenizer
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
logger = logging.getLogger(__name__)
"""
建立网络模型结构
"""

class BertSentenceEncoder(nn.Module):
    def __init__(self, config):
        super(BertSentenceEncoder, self).__init__()
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"],return_dict=False)
        self.hidden_size = self.bert.config.hidden_size
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.layer = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(0.5)



    #输入为问题字符编码
    # def forward(self, x):
    #     x = self.embedding(x)
    #     #使用lstm
    #     # x, _ = self.lstm(x)
    #     #使用线性层
    #     x = self.layer(x)
    #     x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
    #     return x

    def forward(self, input_ids):
        if not input_ids.is_cuda and next(self.parameters()).is_cuda:
            input_ids = input_ids.cuda()

        # BERT前向传播

        _, pooled_output = self.bert(input_ids)  # 直接获取pooled_output作为句子表示
        pooled_output = self.dropout(pooled_output)
        return torch.nn.functional.normalize(pooled_output, p=2, dim=1)




class BertTripletNetwork(nn.Module):
    def __init__(self, config):
        super(BertTripletNetwork, self).__init__()
        self.sentence_encoder = BertSentenceEncoder(config)
        # Triplet Loss的margin参数
        self.margin = config.get("triplet_margin", 0.2)

        # 损失函数
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.cosine_loss = nn.CosineEmbeddingLoss()

        logger.info(f"初始化BERT模型: {config["pretrain_model_path"]}")
        logger.info(f"Triplet margin: {self.margin}")

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than

    #sentence : (batch_size, max_length)
    # def forward(self, sentence1, sentence2=None, target=None):
    #     #同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         #如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         #如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     #单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)

    def forward(self, sentence1, sentence2=None, sentence3=None, target=None):
        if sentence2 is not None and sentence3 is not None:
            # 三元组模式
            vector1 = self.sentence_encoder(sentence1)  # anchor
            vector2 = self.sentence_encoder(sentence2)  # positive
            vector3 = self.sentence_encoder(sentence3)  # negative
            # 计算Triplet Loss
            vector1 = torch.nn.functional.normalize(vector1, p=2, dim=1)
            vector2 = torch.nn.functional.normalize(vector2, p=2, dim=1)
            vector3 = torch.nn.functional.normalize(vector3, p=2, dim=1)
            return self.triplet_loss(vector1, vector2, vector3)
        elif sentence2 is not None:
            # 双句子模式
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            if target is not None:
                return self.cosine_loss(vector1, vector2, target.squeeze())
            else:
                return self.cosine_distance(vector1, vector2)
        else:
            # 单句子模式
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    # Config = {
    #     "pretrain_model_path":r"D:\PycharmProjects\AI学习预习\week6+语言模型和预训练\bert-base-chinese",
    #     "triplet_margin": 0.2,
    #     "vocab_size": 10,
    #     "max_length": 4
    # }
    model = BertTripletNetwork(Config)
    tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])
    anchor_text = ["今天天气真好", "我喜欢自然语言处理"]
    positive_text = ["天气真不错", "我爱NLP技术"]
    negative_text = ["明天要下雨", "我讨厌数学"]

    # 编码文本（简化版）
    s1 = tokenizer(anchor_text, padding=True, truncation=True, max_length=Config["max_length"], return_tensors="pt")[
        "input_ids"]
    s2 = tokenizer(positive_text, padding=True, truncation=True, max_length=Config["max_length"], return_tensors="pt")[
        "input_ids"]
    s3 = tokenizer(negative_text, padding=True, truncation=True, max_length=Config["max_length"], return_tensors="pt")[
        "input_ids"]

    # 测试三元组模式
    loss = model(s1, s2, s3)
    print(f"Triplet Loss: {loss.item():.4f}")

    # 测试双句子模式
    distance = model(s1, s2)
    print(f"余弦距离: {distance}")

    # 测试单句子模式
    vectors = model(s1)
    print(f"句子向量形状: {vectors.shape}")