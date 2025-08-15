# -*- coding: utf-8 -*-
# 调整 q k v和分类头的参数
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
模型训练主程序
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
    #加载模型
    model = TorchModel()

    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            task_type="TOKEN_CLS",  # NER是token分类任务
            r=32,                   # 增加rank以提升NER性能
            lora_alpha=64,          # 增加alpha
            lora_dropout=0.1,
            # 为NER任务添加更多target_modules，包含输出投影和分类头
            target_modules=["query", "key", "value", "dense", "classifier"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    
    
    model = get_peft_model(model, peft_config)
    # print(model.state_dict().keys())

    if tuning_tactics == "lora_tuning":
        # ✅ 修复关键问题：正确解冻分类头
        logger.info("🔧 开始解冻分类头...")
        
        # 方法1：直接解冻分类头（正确路径）
        try:
            if hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
                logger.info("✅ 成功解冻分类头: model.classifier")
            else:
                logger.warning("❌ 模型没有classifier属性")
        except Exception as e:
            logger.error(f"❌ 解冻分类头失败: {e}")
        
        # 方法2：解冻所有包含'classifier'的模块
        classifier_found = False
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
                classifier_found = True
                logger.info(f"✅ 解冻参数: {name}, shape: {param.shape}")
        
        if not classifier_found:
            logger.warning("❌ 未找到任何classifier参数！")
        
        # 输出可训练参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"📊 参数统计: 总参数 {total_params:,}, 可训练 {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # 列出所有可训练的参数名称
        logger.info("🎯 可训练参数列表:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"   {name}: {param.shape}")

    # 设备选择：优先CUDA，然后CPU（MPS在某些模型上有兼容性问题）
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    elif torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            logger.info("尝试使用Apple MPS加速")
            model = model.to(device)
            # 简单测试MPS是否可用
            test_tensor = torch.randn(2, 3).to(device)
            _ = test_tensor @ test_tensor.T
            logger.info("✅ MPS测试通过")
        except Exception as e:
            logger.warning(f"MPS不兼容，切换到CPU: {e}")
            device = torch.device("cpu")
            model = model.cpu()
    else:
        device = torch.device("cpu")
        logger.info("使用CPU训练")

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
            # 将数据移动到正确的设备
            batch_data = [d.to(device) for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   # NER输入：token序列和对应的标签序列
            
            # 创建attention_mask
            attention_mask = (input_ids != 0).float()
            
            # 对于NER任务，模型会自动计算损失
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  #保存模型权重
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
