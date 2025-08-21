# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import build_base_model, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig


# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"], exist_ok=True)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载基础模型
    base_model = build_base_model()

    # 大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "dense", "out_proj"],
            task_type="TOKEN_CLS",
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    else:
        raise ValueError("未知的tuning_tactics: %s" % tuning_tactics)

    model = get_peft_model(base_model, peft_config)

    # 对于LoRA，使分类头保持可训练（一般LoRA会冻结base权重）
    if tuning_tactics == "lora_tuning":
        # 兼容不同骨干：将名称中包含"classifier"的参数全部解冻
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

    model = model.to(device)

    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    best_metric = -1.0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            batch_data = [d.to(device) for d in batch_data]

            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch_data
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        f1 = evaluator.eval(epoch)
        best_metric = max(best_metric, f1)

    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  # 保存可训练参数
    logger.info("best F1: %.4f" % best_metric)
    return best_metric


def save_tunable_parameters(model, path):
    saved_params = {k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad}
    torch.save(saved_params, path)


if __name__ == "__main__":
    main(Config)


