import torch
from tqdm import tqdm

from model import BERTClassifier
from loader import create_dataloaders
from transformers import AdamW, get_linear_schedule_with_warmup
import config
import evaluate


def train():
    # 初始化组件
    model = BERTClassifier(num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader, valid_loader = create_dataloaders()
    optimizer = AdamW(model.parameters(), lr=config.Config["learning_rate"], no_deprecation_warning=True)
    total_steps = len(train_loader) * config.Config["epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.Config.get("warmup_steps", 100),
        num_training_steps=total_steps
    )

    # 训练循环
    with tqdm(total=config.Config["epoch"]*len(train_loader), desc="Training") as t:
        for epoch in range(config.Config["epoch"]):
            model.train()
            total_loss = 0

            for batch in train_loader:

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask, labels)
                loss = outputs["loss"]
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                t.set_postfix(loss=total_loss / len(train_loader))
                t.update(1)

            # 验证阶段
            model.eval()
            val_accuracy = evaluate.evaluate_model(model, valid_loader, device)

            print(f"Epoch {epoch + 1}/{config.Config['epoch']}")
            print(f"Train Loss: {total_loss / len(train_loader):.4f} | Val Accuracy: {val_accuracy:.4f}")
            # 保存模型
            if (epoch + 1) % 2 == 0:
                torch.save(model.state_dict(), config.Config["model_path"])


if __name__ == "__main__":
    train()
