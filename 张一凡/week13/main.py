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
    model = TorchModel

    # LoRA配置
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        task_type="TOKEN_CLS"  # 修改为Token分类任务
    )

    model = get_peft_model(model, peft_config)

    # 确保分类层可训练
    for param in model.get_submodule("model").get_submodule("classifier").parameters():
        param.requires_grad = True

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch_data

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        f1 = evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "%s.pth" % config["tuning_tactics"])
    save_tunable_parameters(model, model_path)
    return f1


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


if __name__ == "__main__":
    main(Config)
