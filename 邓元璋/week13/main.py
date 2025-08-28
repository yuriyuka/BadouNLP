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

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    train_data = load_data(config["train_data_path"], config)

    # 初始化模型和LoRA配置
    model = TorchModel
    if config["tuning_tactics"] == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],  # BERT注意力层
            task_type="TOKEN_CLASSIFICATION"  # 明确任务类型
        )
        model = get_peft_model(model, peft_config)
        # 确保分类头（classifier）可训练
        for param in model.classifier.parameters():
            param.requires_grad = True

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
        logger.info("使用GPU训练")

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"epoch {epoch + 1} 开始")
        total_loss = 0.0
        for batch in train_data:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch  # NER的输入包含mask和标签序列
            if cuda_flag:
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

            # 前向传播：计算损失（自动忽略labels=-100的位置）
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        logger.info(f"epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
        evaluator.eval(epoch + 1)  # 每轮结束评估

    # 保存LoRA权重
    model_path = os.path.join(config["model_path"], "lora_ner.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型保存至 {model_path}")


if __name__ == "__main__":
    main(Config)
