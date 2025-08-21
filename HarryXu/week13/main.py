# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
加入LoRA微调
"""

def setup_lora(model, config):
    """配置并应用LoRA适配器"""
    # LoRA配置
    lora_config = LoraConfig(
        r=config["lora_r"],  # 秩
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],  # 需要应用LoRA的模块
        lora_dropout=config["lora_dropout"],
        bias="none",  # 不训练偏置
        task_type="SEQ_CLASSIFICATION" if config["task_type"] == "classification" else "SEQ2SEQ_LM",
    )
    
    # 准备模型（如果使用INT8量化）
    if config["use_int8_training"]:
        model =  TorchModel(config)
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    
    # 加载基础模型
    model = TorchModel(config)
    
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    
    # 应用LoRA适配器
    if config["use_lora"]:
        logger.info("应用LoRA适配器进行微调")
        model = setup_lora(model, config)
    
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
            
            input_id, labels = batch_data
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    
    if config["use_lora"]:
        model_path = os.path.join(config["model_path"], "lora_epoch_%d" % epoch)
        model.save_pretrained(model_path)
        logger.info(f"LoRA适配器参数已保存至 {model_path}")
    else:
        model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        torch.save(model.state_dict(), model_path)
    
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
