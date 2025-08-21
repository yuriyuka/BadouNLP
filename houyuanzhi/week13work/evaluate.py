# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def eval(self, epoch):
        self.model.eval()
        self.stats_dict = {"tp":0, "fp":0, "fn":0}  # 使用实体级别统计
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, attention_mask, labels = batch_data
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=2)
            
            self.write_stats(labels, predictions, attention_mask)
        f1 = self.calculate_f1()
        return f1

    def write_stats(self, labels, predictions, attention_mask):
        # 统计TP/FP/FN
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        mask = attention_mask.view(-1).bool()
        
        # 只计算有效token
        valid_preds = predictions[mask]
        valid_labels = labels[mask]
        
        # 计算实体级别指标
        # 需要实现实体识别逻辑
        entities_pred = self.decode_entities(valid_preds)
        entities_true = self.decode_entities(valid_labels)
        
        # 统计TP/FP/FN
        self.stats_dict["tp"] += len(entities_pred & entities_true)
        self.stats_dict["fp"] += len(entities_pred - entities_true)
        self.stats_dict["fn"] += len(entities_true - entities_true)

    def decode_entities(self, labels):
        # 实现实体解码逻辑
        # 例如将标签序列转换为实体集合
        pass
