# 基于LoRA的命名实体识别(NER)训练代码
# 环境依赖：transformers datasets peft accelerate evaluate seqeval

import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from evaluate import load
import bitsandbytes as bnb

# ------------------------------
# 1. 配置参数
# ------------------------------
MODEL_NAME = "bert-base-chinese"  # 基础预训练模型
DATA_PATH = "./ner_data"  # 数据集路径（需包含train.txt, dev.txt）
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]  # NER标签
LORA_R = 8  # LoRA低秩矩阵维度
LORA_ALPHA = 32  # LoRA缩放因子
LORA_DROPOUT = 0.05
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 3e-4  # LoRA通常使用较大学习率
NUM_EPOCHS = 10
OUTPUT_DIR = "./lora_ner_results"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# 2. 数据处理
# ------------------------------
def load_custom_ner_data(data_dir):
    """加载自定义NER数据集（CoNLL格式：每行"字 标签"，空行分隔句子）"""
    datasets = {}
    for split in ["train", "dev"]:
        file_path = os.path.join(data_dir, f"{split}.txt")
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 空行表示句子结束
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    char, label = line.split()
                    current_sentence.append(char)
                    current_labels.append(label)

        # 转换为Dataset格式
        datasets[split] = {
            "tokens": sentences,
            "labels": labels
        }

    return DatasetDict(datasets)


# 加载数据并映射标签到ID
dataset = load_custom_ner_data(DATA_PATH)
label_to_id = {label: i for i, label in enumerate(LABEL_LIST)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(examples):
    """分词并对齐标签（处理子词拆分问题）"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,  # 输入已按字拆分
        padding="max_length",
        truncation=True,
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # 记录每个token属于哪个原始词
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:  # 特殊token（如[CLS]、[SEP]）
                label_ids.append(-100)  # 忽略这些位置的损失
            elif word_idx != previous_word_idx:  # 原始词的第一个子词
                label_ids.append(label_to_id[label[word_idx]])
            else:  # 同一词的其他子词（继承主标签）
                label_ids.append(label_to_id[label[word_idx]] if label[word_idx].startswith("I-") else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 应用分词和标签对齐
tokenized_dataset = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)


# ------------------------------
# 3. 配置LoRA模型
# ------------------------------
def find_all_linear_names(model):
    """找到所有线性层名称（用于指定LoRA微调目标）"""
    cls = bnb.nn.Linear4bit if getattr(model, 'is_loaded_in_4bit', False) else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 排除输出层
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# 加载基础模型（4-bit量化节省内存）
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    load_in_4bit=True,
    device_map="auto",
    quantization_config=bnb.QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
)

# 准备模型用于k-bit训练
model = prepare_model_for_kbit_training(model)

# 配置LoRA
modules = find_all_linear_names(model)  # 自动发现可微调的线性层
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=modules,  # 对所有线性层应用LoRA
    lora_dropout=LORA_DROPOUT,
    bias="none",  # 不微调偏置
    task_type="TOKEN_CLASSIFICATION",
)

# 应用LoRA包装器
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例（通常<1%）

# ------------------------------
# 4. 训练配置
# ------------------------------
# 加载评估指标（seqeval用于NER评估）
metric = load("seqeval")


def compute_metrics(p):
    """计算NER评估指标（精确率、召回率、F1）"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 收集真实标签和预测标签（过滤-100）
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 数据整理器（处理批次内padding和注意力掩码）
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",  # 不使用wandb等日志工具
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ------------------------------
# 5. 开始训练
# ------------------------------
print("开始训练...")
trainer.train()

# 保存LoRA权重（仅几MB）
model.save_pretrained(f"{OUTPUT_DIR}/lora_weights")
print(f"LoRA权重已保存至 {OUTPUT_DIR}/lora_weights")


# ------------------------------
# 6. 模型推理示例
# ------------------------------
def predict_ner(text):
    """使用训练好的模型进行NER预测"""
    # 分词
    inputs = tokenizer(
        list(text),  # 按字拆分
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to("cuda")

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 解析预测结果
    predictions = torch.argmax(outputs.logits, dim=2)
    word_ids = inputs.word_ids(batch_index=0)

    entities = []
    current_entity = None
    current_label = None

    for i, (word_id, pred_id) in enumerate(zip(word_ids, predictions[0])):
        if word_id is None:
            continue  # 跳过特殊token

        pred_label = id_to_label[pred_id.item()]
        if pred_label.startswith("B-"):
            # 新实体开始
            if current_entity:
                entities.append((current_entity, current_label))
            current_entity = text[word_id]
            current_label = pred_label.split("-")[1]
        elif pred_label.startswith("I-") and current_entity:
            # 实体延续
            current_entity += text[word_id]
        else:
            # 实体结束
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = None
                current_label = None

    # 处理最后一个实体
    if current_entity:
        entities.append((current_entity, current_label))

    return entities


# 测试推理
test_text = "张三在北京大学工作，住在北京市海淀区"
print(f"\n测试文本：{test_text}")
print("预测实体：", predict_ner(test_text))
