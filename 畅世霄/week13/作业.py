Config = {
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "pooling_style":"max",
    "model_path": "output",
    "max_length": 128,
    "epoch": 3,
    "batch_size": 16,
    "tuning_tactics": "lora_tuning",
    "optimizer": "adam",
    "learning_rate": 5e-5,
    "pretrain_model_path": r"E:\PycharmProjects\NLPtask\bert-base-chinese",
    "seed": 42,
    "train_data_path": "train",
    "valid_data_path": "dev",
    "label_map_path": "schema.json",
}
----------------------
import torch
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果：")
        self.model.eval()
        correct, total = 0, 0
        for batch in self.valid_data:
            batch = {k: v.cuda() for k, v in batch.items()} if torch.cuda.is_available() else batch
            with torch.no_grad():
                outputs = self.model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
        acc = correct / total if total > 0 else 0
        self.logger.info(f"NER token-level 准确率: {acc:.4f}")
        return acc
--------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json

class NERDataset(Dataset):
    """读取 CoNLL 格式 NER 数据"""
    def __init__(self, data_path, config):
        self.config = config
        with open(config["label_map_path"], "r", encoding="utf-8") as f:
            self.label2id = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"], use_fast=True)
        self.samples = self._read_conll(data_path)

    def _read_conll(self, path):
        sentences, labels = [], []
        tokens, tags = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        sentences.append(tokens)
                        labels.append(tags)
                        tokens, tags = [], []
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                token, tag = parts
                tokens.append(token)
                tags.append(tag)
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
        return [{"tokens": tks, "labels": lbs} for tks, lbs in zip(sentences, labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]
        labels = self.samples[idx]["labels"]

        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.config["max_length"],
                                  return_tensors="pt")
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(self.label2id[labels[word_id]])

        encoding["labels"] = torch.tensor(label_ids)
        return {k: v.squeeze(0) for k, v in encoding.items()}

def load_data(data_path, config, shuffle=True):
    dataset = NERDataset(data_path, config)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
---------------------
import json
from config import Config
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import Adam, SGD

# 读取标签映射
with open(Config["label_map_path"], "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# 加载 Token Classification 模型
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["pretrain_model_path"],
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)


TorchModel.config.return_dict = True

# 分词器
Tokenizer = AutoTokenizer.from_pretrained(Config["pretrain_model_path"], use_fast=True)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer.lower() == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
--------------
import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from evaluate import Evaluator
from config import Config


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s"%tuning_tactics)

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]
    )
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)

#重建模型
model = TorchModel
# print(model.state_dict().keys())
# print("====================")

model = get_peft_model(model, peft_config)
# print(model.state_dict().keys())
# print("====================")

state_dict = model.state_dict()

#将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('output/lora_tuning.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('output/prefix_tuning.pth')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

#权重更新后重新加载到模型
model.load_state_dict(state_dict)

#进行一次测试
model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)
----------------
import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 固定随机种子
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

    # LoRA 配置
    if config["tuning_tactics"] == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
        model = get_peft_model(model, peft_config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("使用 GPU")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"epoch {epoch+1} begin")
        train_loss = []
        for batch in train_data:
            batch = {k: v.cuda() for k, v in batch.items()} if cuda_flag else batch
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                outputs.logits.view(-1, outputs.logits.shape[-1]),
                batch["labels"].view(-1)
            )
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        logger.info(f"epoch {epoch+1} loss: {np.mean(train_loss):.4f}")
        evaluator.eval(epoch+1)

    model_path = os.path.join(config["model_path"], f"{config['tuning_tactics']}.pth")
    save_tunable_parameters(model, model_path)

def save_tunable_parameters(model, path):
    saved_params = {k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad}
    torch.save(saved_params, path)

if __name__ == "__main__":
    main(Config)
