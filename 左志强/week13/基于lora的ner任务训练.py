
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    TaskType
)
from datasets import load_dataset
import evaluate
import numpy as np

# 参数配置
MODEL_NAME = "bert-base-cased"  # 基础模型
DATASET_NAME = "conll2003"      # 数据集
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
LORA_R = 8  # LoRA秩
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0.1
OUTPUT_DIR = "lora_ner_model"

# 加载数据集
dataset = load_dataset(DATASET_NAME)

# 标签转换
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 数据预处理函数
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
            # 当前单词的第一个子词
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            # 同一单词的后续子词
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 处理数据集
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 初始化基础模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 配置LoRA
peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "key", "value"]  # 针对BERT的注意力层
)

# 应用LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数占比

# 数据收集器
data_collator = DataCollatorForTokenClassification(tokenizer)

# 评估指标
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, 
        references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# 训练配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存最终模型
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"训练完成! 模型已保存到 {OUTPUT_DIR}")
