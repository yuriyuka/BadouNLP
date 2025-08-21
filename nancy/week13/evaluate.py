# -*- coding: utf-8 -*-
import torch
import numpy as np
from loader import load_data
from seqeval.metrics import precision_score, recall_score, f1_score

"""
模型效果测试
"""

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.id2label = {i: l for i, l in enumerate(config["label_list"])}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for index, batch_data in enumerate(self.valid_data):
                batch_data = [d.to(device) for d in batch_data]
                input_ids, attention_mask, labels = batch_data
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                pred_ids = torch.argmax(logits, dim=-1)
                # 过滤-100的标签
                for true_row, pred_row in zip(labels.cpu().numpy(), pred_ids.cpu().numpy()):
                    true_tags, pred_tags = [], []
                    for t, p in zip(true_row, pred_row):
                        if t == -100:
                            continue
                        true_tags.append(self.id2label[int(t)])
                        pred_tags.append(self.id2label[int(p)])
                    all_true.append(true_tags)
                    all_pred.append(pred_tags)

        precision = precision_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        self.logger.info("P=%.4f, R=%.4f, F1=%.4f" % (precision, recall, f1))
        self.logger.info("--------------------")
        return f1


