import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# 示例：确认 evaluate_model 中是否正确计算准确率
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs["logits"], dim=1)  # 确保使用 logits
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
