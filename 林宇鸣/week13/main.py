# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
# 第三方库
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)

    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        # 示范，需要设置loraconfig
        peft_config = LoraConfig(
            r=8,  # 映射到8维，r过小的话对模型起不到作用
            lora_alpha=32,  # 放缩参数 可以不管
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]  # 决定在什么地方用lora
            # 每个线性层旁边都可以加lora Wx + Ax + b
            # 大模型输出+lora输出， W_0x + BAx
            # 这个例子考虑加在qkv上 正则表达式找，名字有query key value
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

    # 对原来的模型 套上参数，加上config，就知道要用lora了
    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())
    # # 冻结所有的其它层
    # if tuning_tactics == "lora_tuning":
    #     # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
    #     # 文本分类，bert+线性层，希望线性层仍然照常训练
    #     # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
    #     for param in model.get_submodule("model").get_submodule("classifier").parameters():
    #         param.requires_grad = True  # 设置线性层正常训练

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
                # batch_data 是一个包含 input_ids 和 labels 的列表
            input_id, labels = batch_data  # 现在只解包两个元素
            # attention_mask = (input_ids != 0).long()  # 根据 input_ids 生成 attention_mask

            # input_ids, attention_mask, labels = batch_data  # 确保返回所有需要的内容
            # input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # loss = model(input_id, labels)
            # 获取模型输出
            # outputs = TorchModel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # 解包损失和 logits
            # loss = outputs.loss  # 获取损失
            # logits = outputs.logits  # 获取 logits
            # 获取模型输出
            # outputs = TorchModel(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # save_tunable_parameters(model, model_path)  # 保存模型权重
    torch.save(model.state_dict(), model_path)
    return model, train_data


# def save_tunable_parameters(model, path):
#     saved_params = {
#         k: v.to("cpu")
#         for k, v in model.named_parameters()
#         if v.requires_grad
#     }
#     torch.save(saved_params, path)


if __name__ == "__main__":
    model, train_data = main(Config)