# -*- coding: utf-8 -*-
import torch
from loader import load_data, load_label_map
from seqeval.metrics import classification_report, f1_score

"""
模型效果测试（适配NER任务）
"""

# 加载标签映射（从schema.json）
label_to_index, index_to_label = load_label_map()
id2label = {v: k for k, v in label_to_index.items()}

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.all_true_labels = []  # 存储所有真实标签序列
        self.all_pred_labels = []  # 存储所有预测标签序列

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.all_true_labels.clear()  # 清空上一轮结果
        self.all_pred_labels.clear()

        for batch_data in self.valid_data:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]

            # 解包3个元素（input_ids, attention_mask, labels）
            input_ids, attention_mask, labels = batch_data

            with torch.no_grad():
                # 关键修改：模型输出为元组时，按索引提取logits（通常是第0个元素）
                model_outputs = self.model(input_ids, attention_mask=attention_mask)
                pred_results = model_outputs[0]  # 从元组中获取logits

            self.write_stats(labels, pred_results, attention_mask)

        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results, attention_mask):
        # 转换为numpy数组并迁移到CPU
        pred_labels = torch.argmax(pred_results, dim=2).cpu().numpy()
        true_labels = labels.cpu().numpy()
        masks = attention_mask.cpu().numpy()

        # 遍历每个样本
        for true_seq, pred_seq, mask in zip(true_labels, pred_labels, masks):
            # 过滤padding（mask=0）和特殊符号（label=-100）
            valid_indices = (mask == 1) & (true_seq != -100)
            # 提取有效标签并转换为文本
            true_valid = [id2label[true_seq[i]] for i in range(len(true_seq)) if valid_indices[i]]
            pred_valid = [id2label[pred_seq[i]] for i in range(len(pred_seq)) if valid_indices[i]]
            # 收集结果
            self.all_true_labels.append(true_valid)
            self.all_pred_labels.append(pred_valid)

    def show_stats(self):
        # 计算NER评估指标
        report = classification_report(self.all_true_labels, self.all_pred_labels, digits=4)
        f1 = f1_score(self.all_true_labels, self.all_pred_labels)
        total = sum(len(seq) for seq in self.all_true_labels)
        correct = sum(
            sum(t == p for t, p in zip(true, pred))
            for true, pred in zip(self.all_true_labels, self.all_pred_labels)
        )
        acc = correct / total if total > 0 else 0

        self.logger.info("预测集合token总量：%d" % total)
        self.logger.info("预测正确token：%d，预测错误token：%d" % (correct, total - correct))
        self.logger.info("token级别准确率：%f" % acc)
        self.logger.info("实体级别评估报告:\n%s" % report)
        self.logger.info("整体F1分数：%f" % f1)
        self.logger.info("--------------------")
        return f1