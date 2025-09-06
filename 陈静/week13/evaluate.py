# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  #存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids)[0]
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        if self.config.get("task_type") == "ner":
            # NER任务：token级别的评估
            for batch_labels, batch_pred in zip(labels, pred_results):
                # batch_pred shape: (seq_len, num_labels)
                batch_pred = torch.argmax(batch_pred, dim=-1)  # (seq_len,)
                
                # 只计算非padding位置的准确率
                for true_label, pred_label in zip(batch_labels, batch_pred):
                    if true_label != 0:  # 忽略padding位置 (标签0通常是'O'或padding)
                        if int(true_label) == int(pred_label):
                            self.stats_dict["correct"] += 1
                        else:
                            self.stats_dict["wrong"] += 1
        else:
            # 原有的分类任务逻辑
            for true_label, pred_label in zip(labels, pred_results):
                pred_label = torch.argmax(pred_label)
                if int(true_label) == int(pred_label):
                    self.stats_dict["correct"] += 1
                else:
                    self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
