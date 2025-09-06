# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
NER模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0, "total": 0}

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0, "total": 0}

        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, attention_mask, labels = batch_data
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)

            self.write_stats(labels, preds, attention_mask)

        acc, precision, recall, f1 = self.show_stats()
        return f1

    def write_stats(self, labels, preds, attention_mask):
        # Only evaluate on non-padded tokens
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if attention_mask[i][j] == 1 and labels[i][j] != -100:
                    self.stats_dict["total"] += 1
                    if labels[i][j] == preds[i][j]:
                        self.stats_dict["correct"] += 1
                    else:
                        self.stats_dict["wrong"] += 1

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = self.stats_dict["total"]

        acc = correct / total if total > 0 else 0
        precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0
        recall = correct / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        self.logger.info("预测token总量：%d" % total)
        self.logger.info("预测正确token：%d，预测错误token：%d" % (correct, wrong))
        self.logger.info("准确率：%f" % acc)
        self.logger.info("精确率：%f" % precision)
        self.logger.info("召回率：%f" % recall)
        self.logger.info("F1分数：%f" % f1)
        self.logger.info("--------------------")
        return acc, precision, recall, f1
