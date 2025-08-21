# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertModel, BertTokenizer
from config import Config
from loader import load_data
import logging
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from evaluate import Evaluator

"""
基于pytorch的BERT语言模型 peft微调
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(LanguageModel, self).__init__()
        # attn_implementation 可以使mask 为3维传入
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False,
                                              attn_implementation='eager',
                                              num_hidden_layers=4)
        input_dim = self.bert.config.hidden_size
        self.classify = nn.Linear(input_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
        # self.loss = nn.functional.cross_entropy
        # 设置忽略标签值为 -100 的样本，使其不对损失计算产生影响。常用于处理变长序列中填充（padding）部分的标签
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # if torch.cuda.is_available():
            #     mask = mask.cuda()
            mask = self.build_mask(x)
            # 增加 batch_size 维度：(1, seq_len, seq_len)
            mask = mask.unsqueeze(0)  # shape: (1, 5, 5)
            # 扩展为 (batch_size, seq_len, seq_len)
            mask = mask.expand(x.shape[0], -1, -1)  # shape: (4, 5, 5)
            x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim) 64 10 256
            x = self.dropout(x)  # 32 150 768
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时 可以不使用mask
            x, _ = self.bert(x)
            x = self.dropout(x)
            y_pred = self.classify(x)  # output shape:(batch_size, sen_len, vocab_size) 64 10 3961
            return torch.softmax(y_pred, dim=-1)

    def build_mask(self, x):
        x_len = x.shape[1]
        # 查找 102 的位置
        sep_positions = (x == 102).nonzero(as_tuple=True)[1]
        # print(sep_positions)  # 输出: tensor([3, 5])
        mask = torch.zeros(x_len, x_len, dtype=torch.float32)
        # print(mask)
        for i in range(x_len):
            for j in range(x_len):
                if i < sep_positions[0] and j < sep_positions[0]:
                    mask[i][j] = 1  # 前 sep_index 行列全为 1
                elif j <= i:
                    mask[i][j] = 1  # 后续部分为下三角矩阵
        # print(mask)
        return mask


# 建立模型
def build_model(vocab_size, config):
    model = LanguageModel(vocab_size, config)
    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        # Low-Rank Adaptation 低秩适应
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)
    if tuning_tactics == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("classify").parameters():
            param.requires_grad = True
    return model


def train(corpus_path, config, save_weight=True):
    epoch_num = config["epoch"]  # 训练轮数
    tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
    train_data = load_data(corpus_path, config, logger)
    model = build_model(len(tokenizer.vocab), config)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    # 加载效果测试类
    evaluator = Evaluator(config, model, tokenizer, logger)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_data in train_data:
            input_seq, gold = batch_data
            if torch.cuda.is_available():
                input_seq, gold = input_seq.cuda(), gold.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(input_seq, gold)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluator.eval(epoch + 1)
    if not save_weight:
        return
    else:
        model_path = os.path.join(config["model_path"], "%s.pth" % config["tuning_tactics"])
        save_tunable_parameters(model, model_path)  # 保存模型权重
        return


def save_tunable_parameters(model, path):
    # tunable 可调谐的
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path)


if __name__ == "__main__":
    train("data/sample_data.json", Config, True)

    # # 假设 x 是一个 tensor
    # x = torch.tensor([[101,999, 123, 456, 102, 789, 756, 888]])
    # x_len = x.shape[1]
    # # 查找 102 的位置
    # sep_positions = (x == 102).nonzero(as_tuple=True)[1]
    # print(sep_positions)  # 输出: tensor([3, 5])
    # mask = torch.zeros(x_len, x_len, dtype=torch.float32)
    # # print(mask)
    # for i in range(x_len):
    #     for j in range(x_len):
    #         if i < sep_positions[0] and j < sep_positions[0]:
    #             mask[i][j] = 1  # 前 sep_index 行列全为 1
    #         elif j <= i:
    #             mask[i][j] = 1  # 后续部分为下三角矩阵
    # print(mask)

    # tensor([[1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 0., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 0., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 0.],
    #         [1., 1., 1., 1., 1., 1., 1., 1.]])
