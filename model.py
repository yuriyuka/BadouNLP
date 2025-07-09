# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=0.1):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        diff = ap - an + margin
        return torch.mean(torch.relu(diff)) # 使用ReLU确保只有正的损失被计算

    #sentence : (batch_size, max_length)
    def forward(self, *args):
        # 同时传入三个句子时，计算triplet loss
        if len(args) == 3:  # 三元组情况: anchor, positive, negative
            anchor, positive, negative = args
            v_a = self.sentence_encoder(anchor)
            v_p = self.sentence_encoder(positive)
            v_n = self.sentence_encoder(negative)
            return self.cosine_triplet_loss(v_a, v_p, v_n)
        #同时传入两个句子和标签时，计算二元分类损失
        elif len(args) == 3 and isinstance(args[2], torch.Tensor):  # args[2] is target
            sentence1, sentence2, target = args
            vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            distance = self.cosine_distance(vector1, vector2)
            # 将target从[-1,1]转换为[0,1]用于计算二元交叉熵
            target = (target + 1) / 2
            # 确保target是一维的，与distance大小匹配
            if target.dim() > 1:
                target = target.squeeze()  # 移除多余的维度
            return torch.nn.functional.binary_cross_entropy_with_logits(distance, target)
        #单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(args[0])

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    #测试三元组
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0], [3, 3, 0, 0]])  # anchor
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4], [4, 4, 0, 0]])  # positive
    s3 = torch.LongTensor([[2, 3, 4, 0], [4, 3, 2, 0], [5, 5, 0, 0]])  # negative
    # 将三个句子组合成一个批次
    anchor = s1
    positive = s2
    negative = s3
    # 计算triplet loss
    v_a = model.sentence_encoder(anchor)
    v_p = model.sentence_encoder(positive)
    v_n = model.sentence_encoder(negative)
    loss = model.cosine_triplet_loss(v_a, v_p, v_n)
    print("Triplet loss:", loss.item())
    # 测试二元分类
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])  # 1表示相似，0表示不相似
    y = model(s1, s2, l)
    print("Binary classification loss:", y.item())