import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification, get_scheduler
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 加载数据集
dataset = load_dataset('conll2003')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# 数据预处理
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8)

# 加载模型并应用LoRA
model_name_or_path = "bert-base-cased"
num_labels = len(dataset['train'].features[f"ner_tags"].feature.names)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = BertForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
model = get_peft_model(model, lora_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 训练循环
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    progress_bar = tqdm(range(len(train_dataloader)))
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 评估循环
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        correct_predictions += (predictions == batch['labels']).sum().item()
        total_predictions += (batch['labels'] != -100).sum().item()

print(f"Accuracy: {correct_predictions / total_predictions:.4f}")

# 保存模型
model.save_pretrained("./lora_bert_ner")
tokenizer.save_pretrained("./lora_bert_ner")
