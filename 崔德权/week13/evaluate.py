# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from loader import load_data

"""
NER模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.id2label = config["id2label"]
        self.label2id = config["label2id"]

    def eval(self, epoch, valid_data):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for index, batch_data in enumerate(valid_data):
                if torch.cuda.is_available():
                    batch_data = {k: v.cuda() for k, v in batch_data.items()}
                
                outputs = self.model(**batch_data)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                # 收集预测结果和真实标签
                for pred, label in zip(predictions.cpu().numpy(), batch_data["labels"].cpu().numpy()):
                    # 只考虑非-100的标签（即有效的token）
                    valid_mask = label != -100
                    valid_preds = pred[valid_mask]
                    valid_labels = label[valid_mask]
                    
                    all_predictions.extend(valid_preds)
                    all_labels.extend(valid_labels)
        
        # 计算评估指标
        accuracy = self.calculate_accuracy(all_labels, all_predictions)
        precision, recall, f1 = self.calculate_ner_metrics(all_labels, all_predictions)
        
        self.logger.info(f"准确率: {accuracy:.4f}")
        self.logger.info(f"精确率: {precision:.4f}")
        self.logger.info(f"召回率: {recall:.4f}")
        self.logger.info(f"F1分数: {f1:.4f}")
        self.logger.info("--------------------")
        
        return f1  # 返回F1分数作为主要指标

    def calculate_accuracy(self, labels, predictions):
        """计算准确率"""
        correct = sum(1 for l, p in zip(labels, predictions) if l == p)
        return correct / len(labels) if labels else 0

    def calculate_ner_metrics(self, labels, predictions):
        """计算NER任务的精确率、召回率和F1分数"""
        # 过滤掉O标签，只计算实体标签的指标
        entity_labels = [l for l in labels if l != 0]  # 0是O标签
        entity_predictions = [p for l, p in zip(labels, predictions) if l != 0]
        
        if not entity_labels:
            return 0.0, 0.0, 0.0
        
        # 计算精确率、召回率、F1分数
        precision = precision_score(entity_labels, entity_predictions, average='weighted', zero_division=0)
        recall = recall_score(entity_labels, entity_predictions, average='weighted', zero_division=0)
        f1 = f1_score(entity_labels, entity_predictions, average='weighted', zero_division=0)
        
        return precision, recall, f1

    def show_detailed_report(self, labels, predictions):
        """显示详细的分类报告"""
        label_names = [self.id2label[i] for i in range(len(self.id2label))]
        report = classification_report(labels, predictions, target_names=label_names, zero_division=0)
        self.logger.info("详细分类报告:")
        self.logger.info(report)
