# -*- coding: utf-8 -*-
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
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
        logger.info("正在使用 LoRA 微调策略")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["classify"]  # 只对分类层应用LoRA
        )
        model = get_peft_model(model, peft_config)

        # 手动设置embedding和CRF层允许训练
        for param in model.get_submodule("base_model").get_submodule("model").get_submodule("embedding").parameters():
            param.requires_grad = True
        for param in model.get_submodule("base_model").get_submodule("model").get_submodule("crf_layer").parameters():
            param.requires_grad = True

        lstm_layer = model.get_submodule("base_model").get_submodule("model").get_submodule("layer")
        for name, param in lstm_layer.named_parameters():
            # 只训练LTSM最后一层的参数
            if "weight_hh_l1" in name or "weight_ih_l1" in name or "bias_hh_l1" in name or "bias_ih_l1" in name:
                param.requires_grad = True

        logger.info("LoRA配置完成，训练embedding、部分LSTM层、分类层(LoRA)和CRF层")
    else:
        logger.info("使用全参微调")

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
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

    # 保存模型
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    if tuning_tactics == "lora_tuning":
        save_tunable_parameters(model, model_path)  # 只保存可训练参数
        logger.info(f"LoRA parameters saved to {model_path}")
    else:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Full model saved to {model_path}")
    return model, train_data

def save_tunable_parameters(model, path):
    """保存可训练参数"""
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
    logger.info(f"Saved {len(saved_params)} tunable parameters")

if __name__ == "__main__":
    model, train_data = main(Config)