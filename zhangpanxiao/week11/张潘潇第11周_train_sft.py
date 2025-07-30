import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import os


# 使用新闻数据尝试实现sft训练。 主训练脚本

# 配置参数
MODEL_NAME = "baichuan-inc/Baichuan2-7B-Base"  # 可替换为其他预训练模型
DATA_PATH = "data/processed/sft_dataset"
OUTPUT_DIR = "results/sft_news"
MAX_LENGTH = 512  # 根据显存调整

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token


def format_prompt(example):
    """构造提示模板"""
    prompt = f"指令:{example['instruction']}\n输入:{example['input']}\n输出:"
    return {"prompt": prompt}


def tokenize_function(examples):
    """分词函数"""
    texts = [format_prompt(ex)["prompt"] + ex["output"] for ex in examples]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    # 创建标签（掩码输入部分）
    labels = tokenized["input_ids"].copy()
    prompt_lens = [len(tokenizer(format_prompt(ex)["prompt"]) for ex in examples)]

    for i, length in enumerate(prompt_lens):
        labels[i][:length] = [-100] * length  # 忽略输入部分的损失

    return {"input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels}


def main():
    # 加载数据集
    dataset = load_from_disk(DATA_PATH)
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )

    # 分割训练/验证集
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # 根据GPU调整
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,  # 使用混合精度
        gradient_accumulation_steps=4,  # 梯度累积
        report_to="tensorboard"
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
    )

    # 开始训练
    print("开始SFT训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))


if __name__ == "__main__":
    main()
