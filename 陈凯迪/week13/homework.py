import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# 1. 加载数据集
dataset = load_dataset("conll2003")

# 2. 预处理数据
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 定义标签映射
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        padding="max_length"
    )

    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 特殊token设置为-100
            if word_idx is None:
                label_ids.append(-100)
            # 同一个词的其他子词
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 3. 创建LoRA模型
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,  # 令牌分类任务
    inference_mode=False,
    r=8,  # LoRA秩
    lora_alpha=32,  # 缩放因子
    lora_dropout=0.1,  # Dropout率
    target_modules=["query", "key", "value", "dense"]  # 目标模块
)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 查看可训练参数比例

# 4. 训练配置
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="lora-ner-results",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"
)


# 5. 修正后的评估函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除忽略的标签（-100）
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        preds_list = []
        labels_list = []
        for p, l in zip(prediction, label):
            if l != -100:
                preds_list.append(label_list[p])
                labels_list.append(label_list[l])
        true_predictions.append(preds_list)
        true_labels.append(labels_list)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy": (np.array(predictions) == np.array(labels)).mean(),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 6. 训练模型
trainer.train()

# 7. 保存模型
model.save_pretrained("lora-ner-model")
