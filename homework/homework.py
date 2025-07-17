# 安装必要库（如果未安装）
# pip install transformers torch datasets seqeval

import json
import os

import numpy as np
import torch
from datasets import Dataset as HFDataset, load_dataset
from seqeval.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 加载本地数据集 - 支持两种格式：CONLL格式和自定义JSON格式
def load_local_dataset(data_dir, dataset_format="conll"):
    """
    加载本地数据集
    :param data_dir: 数据目录路径
    :param dataset_format: 数据集格式 ('conll' 或 'json')
    :return: Hugging Face DatasetDict对象
    """
    train_path = os.path.join(data_dir, "train")
    valid_path = os.path.join(data_dir, "test")
    test_path = os.path.join(data_dir, "test")

    if dataset_format == "conll":
        # 加载CONLL格式数据集
        dataset = load_dataset("conll2003", data_files={
            "train": train_path,
            "validation": valid_path,
            "test": test_path
        })
    elif dataset_format == "json":
        # 加载自定义JSON格式数据集
        def read_json_file(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

        train_data = read_json_file(train_path)
        valid_data = read_json_file(valid_path)
        test_data = read_json_file(test_path)

        # 转换为Hugging Face Dataset格式
        dataset = HFDataset.from_dict({
            "train": train_data,
            "validation": valid_data,
            "test": test_data
        })
    else:
        raise ValueError("不支持的格式。请选择 'conll' 或 'json'")

    return dataset


# 使用示例 - 根据您的数据格式选择一种
# dataset = load_local_dataset("data/conll2003", dataset_format="conll")
dataset = load_local_dataset("ner_data", dataset_format="json")

# 2. 定义标签映射
# 如果是自定义数据集，需要从数据中提取所有标签
if "ner_tags" in dataset["train"].features:
    # 如果数据集已经包含ner_tags特征
    label_list = dataset["train"].features["ner_tags"].feature.names
else:
    # 从数据中收集所有标签
    all_labels = set()
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            for tag in item["ner_tags"]:
                all_labels.add(tag)
    label_list = sorted(list(all_labels))

label_map = {i: label for i, label in enumerate(label_list)}
print("Label Map:", label_map)

# 3. 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 4. 数据处理类
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_map, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length
        # 创建标签到ID的映射
        self.label2id = {label: idx for idx, label in enumerate(label_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]

        # 如果标签是字符串形式，转换为ID
        if isinstance(item["ner_tags"][0], str):
            labels = [self.label2id.get(tag, -100) for tag in item["ner_tags"]]
        else:
            labels = item["ner_tags"]

        # 分词并处理标签对齐
        input_ids = []
        label_ids = []

        for i, (token, label_id) in enumerate(zip(tokens, labels)):
            # 分词
            tokenized = tokenizer.tokenize(token)
            input_ids.extend(tokenizer.convert_tokens_to_ids(tokenized))

            # 为子词分配标签：第一个子词保留原标签，其余使用-100（忽略）
            label_ids.extend([label_id] + [-100] * (len(tokenized) - 1))

        # 截断到最大长度（保留[CLS]和[SEP]的位置）
        input_ids = [tokenizer.cls_token_id] + input_ids[:self.max_length - 2] + [tokenizer.sep_token_id]
        label_ids = [-100] + label_ids[:self.max_length - 2] + [-100]

        # 填充
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        if padding_length > 0:
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            label_ids = label_ids + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }


# 5. 创建数据加载器
train_data = NERDataset(dataset["train"], tokenizer, label_map)
valid_data = NERDataset(dataset["validation"], tokenizer, label_map)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)

# 6. 初始化模型
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)
model.to(device)

# 7. 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# 8. 训练函数
def train_model(epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # 数据移到设备
            inputs = {k: v.to(device) for k, v in batch.items()}

            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # 每个epoch结束后评估
        evaluate_model()


# 9. 评估函数
def evaluate_model():
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in valid_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].numpy()

            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()

            # 获取预测标签（忽略填充和特殊token）
            batch_predictions = np.argmax(logits, axis=2)

            # 移除填充位置和特殊token（标签为-100的位置）
            for i in range(len(batch_predictions)):
                mask = labels[i] != -100
                valid_preds = batch_predictions[i][mask]
                valid_labels = labels[i][mask]

                # 转换数字标签为字符串标签
                pred_labels = [label_map[p] for p in valid_preds]
                true_label_texts = [label_map[l] for l in valid_labels]

                predictions.append(pred_labels)
                true_labels.append(true_label_texts)

    # 使用seqeval计算指标
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, zero_division=0))


# 10. 训练并评估模型
if __name__ == "__main__":
    print("Starting training...")
    train_model(epochs=3)

    # 创建保存目录
    save_dir = "bert_ner_model"
    os.makedirs(save_dir, exist_ok=True)

    # 保存模型
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to '{save_dir}' directory")