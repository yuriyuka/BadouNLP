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
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

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
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel
    
    # PEFT Fine-tuning strategy
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        # TaskType changed to TOKEN_CLS for NER
        peft_config = LoraConfig(
            task_type="TOKEN_CLS",
            inference_mode=False,
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics in ["p_tuning", "prompt_tuning", "prefix_tuning"]:
        # Other tuning methods also need TOKEN_CLS task type
        if tuning_tactics == "p_tuning":
            peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prompt_tuning":
            peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        else: # prefix_tuning
            peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
    
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU is available, moving model to GPU")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch} begins")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch_data

            # The model computes the loss internally when labels are provided
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 5) == 0:
                logger.info(f"Batch loss: {loss.item()}")
        
        logger.info(f"Epoch average loss: {np.mean(train_loss)}")
        evaluator.eval(epoch)
    
    model_path = os.path.join(config["model_path"], f"{tuning_tactics}.pth")
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)