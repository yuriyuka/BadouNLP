# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig 


#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
NER模型训练主程序
"""


seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)



def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载验证数据
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)
    #加载模型
    model = TorchModel(config)

    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            task_type="TOKEN_CLASSIFICATION"
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="TOKEN_CLASSIFICATION", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="TOKEN_CLASSIFICATION", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="TOKEN_CLASSIFICATION", num_virtual_tokens=10)
    
    model = get_peft_model(model, peft_config)
    logger.info(f"使用{tuning_tactics}策略进行训练")
    logger.info(f"可训练参数数量: {model.print_trainable_parameters()}")

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
            if cuda_flag:
                batch_data = {k: v.cuda() for k, v in batch_data.items()}

            optimizer.zero_grad()
            outputs = model(**batch_data)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % max(1, int(len(train_data) / 4)) == 0:
                logger.info("batch %d/%d loss %f" % (index, len(train_data), loss.item()))
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch, valid_data)
    
    # 保存模型
    model_path = os.path.join(config["model_path"], "%s_ner.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  #保存模型权重
    logger.info(f"模型已保存到: {model_path}")
    return acc

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


if __name__ == "__main__":
    main(Config)
