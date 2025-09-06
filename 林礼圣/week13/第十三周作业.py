#week13作业
#实现基于lora的ner任务训练。

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 数据准备 (示例数据集 - 实际应用中应替换为真实NER数据集)
class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]
        
        # 分词和标签对齐
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # 创建标签张量
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        offset_mapping = encoding['offset_mapping'][0]
        
        label_ids = torch.zeros(self.max_len, dtype=torch.long)
        label_ids[0] = -100  # [CLS] token
        label_ids[-1] = -100  # [SEP] token
        
        # 对齐标签
        token_labels = []
        char_to_token = defaultdict(lambda: -1)
        
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 特殊token
                continue
            char_to_token[start] = token_idx
        
        for char_idx, label in enumerate(labels):
            token_idx = char_to_token.get(char_idx, -1)
            if token_idx != -1 and token_idx < self.max_len:
                token_labels.append(self.label_map[label])
        
        # 填充标签序列
        for i in range(1, len(input_ids) - 1):
            if i < len(token_labels) + 1:
                label_ids[i] = token_labels[i-1]
            else:
                label_ids[i] = self.label_map['O']  # 默认为'O'
        
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'labels': label_ids.to(device)
        }

# 示例数据
sentences = [
    "Apple is looking to buy U.K. startup for $1 billion",
    "Tim Cook announced new products in California"
]
labels = [
    ["B-ORG", "O", "O", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O"],
    ["B-PER", "I-PER", "O", "O", "O", "O", "B-LOC"]
]

# 2. LoRA 实现
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # 初始化
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        # 冻结原始权重
        self.frozen_weight = nn.Parameter(torch.zeros(out_dim, in_dim), requires_grad=False)
    
    def forward(self, x):
        # 原始权重 + LoRA适配
        weight = self.frozen_weight + self.scaling * (self.lora_B @ self.lora_A)
        return nn.functional.linear(x, weight)

# 3. 基于BERT的NER模型 + LoRA适配
class BERTLoRAForNER(nn.Module):
    def __init__(self, base_model, num_labels, lora_rank=8, lora_alpha=16):
        super().__init__()
        self.bert = base_model
        self.num_labels = num_labels
        
        # 应用LoRA到BERT的注意力机制
        for layer in self.bert.encoder.layer:
            self._replace_layer_with_lora(layer.attention.self.query, lora_rank, lora_alpha)
            self._replace_layer_with_lora(layer.attention.self.key, lora_rank, lora_alpha)
            self._replace_layer_with_lora(layer.attention.self.value, lora_rank, lora_alpha)
        
        # 分类头
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def _replace_layer_with_lora(self, layer, rank, alpha):
        original_weight = layer.weight.data.clone()
        lora_layer = LoRALayer(
            in_dim=layer.in_features,
            out_dim=layer.out_features,
            rank=rank,
            alpha=alpha
        )
        lora_layer.frozen_weight.data = original_weight
        layer.weight = lora_layer.weight  # 替换原始权重
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits

# 4. 训练配置
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased').to(device)

# 创建数据集
dataset = NERDataset(sentences, labels, tokenizer, max_len=32)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 创建LoRA模型
num_labels = len(dataset.label_map)
lora_model = BERTLoRAForNER(model, num_labels, lora_rank=8, lora_alpha=16).to(device)

# 训练参数
optimizer = AdamW(lora_model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 10
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# 5. 训练循环
lora_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        loss, _ = lora_model(input_ids, attention_mask, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# 6. 保存LoRA参数 (仅保存适配器部分)
def get_lora_state_dict(model):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param.data.cpu()
    return lora_state_dict

torch.save(get_lora_state_dict(lora_model), "lora_ner_weights.pth")

# 7. 推理示例
def predict_entities(model, tokenizer, sentence, label_map):
    model.eval()
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        _, logits = lora_model(input_ids, attention_mask)
    
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
    
    entities = []
    current_entity = ""
    current_label = ""
    
    for token, pred in zip(tokens, predictions):
        label = reverse_label_map.get(pred, 'O')
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_label[2:]))
            current_entity = token
            current_label = label
        
        elif label.startswith('I-') and current_label[2:] == label[2:]:
            current_entity += token.replace("##", "")
        
        else:
            if current_entity:
                entities.append((current_entity, current_label[2:]))
            current_entity = ""
            current_label = ""
    
    return entities

# 测试预测
test_sentence = "Microsoft announced new products in Seattle"
entities = predict_entities(lora_model, tokenizer, test_sentence, dataset.label_map)
print(f"Sentence: {test_sentence}")
print("Entities:", entities)
