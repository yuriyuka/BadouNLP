# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
from peft import LoraConfig, get_peft_model
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        max_length = config["max_length"]
        class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        
        # 检查是否有可用的GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False).to(self.device)
        
        # 应用LoRA
        if config.get("use_lora", False):
            lora_config = LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["lora_target_modules"],
                lora_dropout=config["lora_dropout"],
                bias="none",
                task_type="TOKEN_CLS"
            )
            self.bert = get_peft_model(self.bert, lora_config)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, target=None):
        # 确保输入张量在正确的设备上
        input_ids = input_ids.to(self.device)
        if target is not None:
            target = target.to(self.device)
        
        # 尝试直接访问基础BERT模型
        if hasattr(self.bert, 'base_model'):
            base_bert = self.bert.base_model
        else:
            base_bert = self.bert
        
        # 只传递input_ids参数给基础模型
        x, _ = base_bert(input_ids=input_ids)
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    import inspect
    import torch
    
    # 加载配置和模型
    config = Config
    model = TorchModel(config)
    
    # 打印BERT模型类型
    print("BERT模型类型:")
    print(type(model.bert))
    
    # 打印BERT模型所有属性
    print("BERT模型属性:")
    print(dir(model.bert))
    
    # 打印BERT模型的forward方法签名
    print("BERT模型forward方法签名:")
    print(inspect.signature(model.bert.forward))
    
    # 创建测试输入
    input_ids = torch.randint(0, 1000, (1, 10))
    
    # 测试直接调用BERT模型
    try:
        outputs = model.bert(input_ids=input_ids)
        print("直接调用BERT模型成功!")
    except Exception as e:
        print(f"直接调用BERT模型失败: {e}")
    
    # 测试调用整个模型
    try:
        labels = torch.randint(0, config["class_num"], (1, 10))
        loss = model(input_ids, labels)
        print("调用整个模型成功!")
    except Exception as e:
        print(f"调用整个模型失败: {e}")