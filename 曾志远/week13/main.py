# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging

from peft import get_peft_model, LoraConfig
from torch import nn

from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel

    # 模型调整策略
    tuning_tactic = config["tuning_tactics"]

    if tuning_tactic == "lora_tuning":
        peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "key", "value"])

    model = get_peft_model(model, peft_config)

    if tuning_tactic == "lora_tuning":
        # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
        # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  #输入变化时这里需要修改，比如多输入，多输出的情况
            pred = model(input_id)[0]
            loss = nn.CrossEntropyLoss(ignore_index=-1)(pred.view(-1, config["class_num"]), labels.view(-1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
