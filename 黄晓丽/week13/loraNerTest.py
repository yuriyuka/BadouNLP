import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch.optim import Adam, SGD
from transformers.modeling_outputs import TokenClassifierOutput

# TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载预训练模型（如BERT）
        self.pretrained = AutoModel.from_pretrained(config["pretrain_model_path"])
        # 添加序列标注分类头
        self.classifier = nn.Linear(self.pretrained.config.hidden_size, config["num_labels"])

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.pretrained(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

  --------
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
    model = TorchModel(config)

    #大模型微调策略
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            task_type="TOKEN_CLS",  # 修改为TOKEN_CLS用于序列标注
            inference_mode=False,
            bias="none"
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
    
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # print(model.state_dict().keys())

    if tuning_tactics == "lora_tuning":
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
    best_f1 = 0.0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch}/{config['epoch']} 开始")
        total_loss = 0

        for batch_idx, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            # NER任务输入: [input_ids, attention_mask, labels]
            input_ids, attention_mask, labels = batch_data

            optimizer.zero_grad()

            # 前向传播 - 获取每个token的预测
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # 计算损失 - 内置损失函数已处理序列标注
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # 每100个batch打印日志
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_data)} - Loss: {loss.item():.4f}")

        # 计算平均损失
        avg_loss = total_loss / len(train_data)
        logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")

        # 在验证集上评估
        f1_score = evaluator.eval(epoch)

        # 保存最佳模型
        if f1_score > best_f1:
            best_f1 = f1_score
            save_path = os.path.join(config["model_path"], f"best_model_epoch{epoch}_f1{best_f1:.4f}")
            model.save_pretrained(save_path)  # 使用HF格式保存
            logger.info(f"保存最佳模型到 {save_path}, F1: {best_f1:.4f}")

    logger.info(f"训练完成，最佳F1分数: {best_f1:.4f}")
    return best_f1

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


if __name__ == "__main__":
    main(Config)
