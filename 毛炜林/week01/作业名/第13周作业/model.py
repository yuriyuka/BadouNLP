import torch
import torch.nn as nn
from config import Config
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.optim import Adam, SGD
from torchcrf import CRF  # 导入CRF库


class TorchModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载预训练的Token分类模型（BERT等）
        self.bert = AutoModelForTokenClassification.from_pretrained(
            config["pretrain_model_path"],
            num_labels=config["num_labels"]  # 标签数量（需与schema中的标签总数一致）
        )
        # 初始化CRF层（batch_first=True表示输入形状为[batch, seq_len, num_tags]）
        #print("自定义模型中的BERT注意力层：")
        #print(self.bert.bert.encoder.layer[0].attention.self)
        self.crf = CRF(num_tags=config["num_labels"], batch_first=True)
        self.config = config

    def forward(self, input_ids, labels=None):
        """
        前向传播函数（不使用attention_mask）
        :param input_ids: 输入token的ID，形状为[batch_size, seq_len]
        :param labels: 标签序列，形状为[batch_size, seq_len]，仅训练时传入
        :return: 训练时返回损失张量，预测时返回解码后的标签序列（list of list）
        """
        # 获取BERT的输出logits（不传入attention_mask）
        outputs = self.bert(input_ids=input_ids)
        logits = outputs[0]  # 从元组中提取logits（BERT的原始预测分数）

        if labels is not None:
            # 训练模式：计算CRF损失
            # 使用labels生成mask（假设-1是填充标记，过滤填充区域）
            mask = labels.gt(-1)  # 形状：[batch_size, seq_len]，True表示有效token
            crf_loss = -self.crf(emissions=logits, tags=labels, mask=mask, reduction="mean")
            return crf_loss  # 返回损失张量，用于反向传播
        else:
            # 预测模式：使用CRF解码（不使用mask，默认所有token有效）
            pred_tags = self.crf.decode(emissions=logits)  # 形状：[batch_size, seq_len]
            return pred_tags  # 返回解码后的标签序列


# 加载分词器
def get_tokenizer(config):
    return AutoTokenizer.from_pretrained(config["pretrain_model_path"])


# 选择优化器（自动包含CRF层参数）
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支持的优化器：{optimizer}")
