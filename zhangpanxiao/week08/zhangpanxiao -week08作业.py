
#三元组损失实现（支持BERT等语言模型）
python
import torch
import torch.nn as nn
from transformers import AutoModel


class TripletBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, anchor, positive, negative):
        # 获取句子嵌入（使用[CLS]向量）
        anchor_emb = self.bert(**anchor).last_hidden_state[:, 0, :]
        pos_emb = self.bert(**positive).last_hidden_state[:, 0, :]
        neg_emb = self.bert(**negative).last_hidden_state[:, 0, :]

        loss = self.triplet_loss(anchor_emb, pos_emb, neg_emb)
        return loss, (anchor_emb, pos_emb, neg_emb)


from torch.utils.data import Dataset
import random

#动态三元组采样器
class TripletTextDataset(Dataset):
    def __init__(self, texts, pairs, tokenizer, max_len=128):
        """
        texts: {id: text}
        pairs: [(anchor_id, positive_id)] 正样本对列表
        """
        self.texts = texts
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_ids = list(texts.keys())

        # 建立反向映射
        self.text_to_positives = {}
        for a, p in pairs:
            self.text_to_positives.setdefault(a, []).append(p)

    def __getitem__(self, idx):
        anchor_id, pos_id = self.pairs[idx]
        neg_id = self._sample_negative(anchor_id)

        return {
            "anchor": self._tokenize(self.texts[anchor_id]),
            "positive": self._tokenize(self.texts[pos_id]),
            "negative": self._tokenize(self.texts[neg_id])
        }

    def _sample_negative(self, anchor_id):
        """困难负样本采样策略"""
        candidates = set(self.text_ids) - set(self.text_to_positives.get(anchor_id, []))
        return random.choice(list(candidates))

    def _tokenize(self, text):
        return self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


from transformers import AdamW, get_linear_schedule_with_warmup


# 高级训练策略
# 1. 困难负样本挖掘（在线生成）
def mine_hard_negatives(model, dataloader, device, k=5):
    model.eval()
    hard_negatives = {}

    with torch.no_grad():
        for batch in dataloader:
            anchor_emb = model.bert(
                batch["anchor"].to(device)
            ).last_hidden_state[:, 0, :]

            # 计算与所有候选样本的相似度
            all_texts = [{"input_ids": ..., "attention_mask": ...}]  # 所有文本tokenized
            all_embs = model.bert(all_texts).last_hidden_state[:, 0, :]

            similarities = torch.matmul(anchor_emb, all_embs.T)

            # 排除正样本
            pos_ids = batch["positive_ids"]
            similarities[:, pos_ids] = -1

            # 取相似度最高的k个负样本
            _, topk_indices = torch.topk(similarities, k=k)
            hard_negatives.update(zip(batch["anchor_ids"], topk_indices.tolist()))

    return hard_negatives

# 2. 带温度系数的对比损失
class TemperatureScaledTripletLoss(nn.Module):
    def __init__(self, margin=1.0, temp=0.05):
        super().__init__()
        self.margin = margin
        self.temp = temp

    def forward(self, anchor, positive, negative):
        pos_sim = torch.cosine_similarity(anchor, positive, dim=-1) / self.temp
        neg_sim = torch.cosine_similarity(anchor, negative, dim=-1) / self.temp

        loss = -torch.log(torch.exp(pos_sim) /
                          (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()

# 完整训练流程
def train():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletBERT().to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 数据准备
    texts = {...}  # {id: text}
    pairs = [...]  # [(id1, id2)] 正样本对
    dataset = TripletTextDataset(texts, pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=1000
    )

    # 训练循环
    for epoch in range(5):
        model.train()
        total_loss = 0

        for batch in dataloader:
            optimizer.zero_grad()

            loss, _ = model(
                batch["anchor"].to(device),
                batch["positive"].to(device),
                batch["negative"].to(device)
            )

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")

        # 每轮进行困难负样本挖掘
        hard_negs = mine_hard_negatives(model, dataloader, device)
        dataset.update_negatives(hard_negs)
