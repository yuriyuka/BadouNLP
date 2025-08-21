# -*- coding: utf-8 -*-
"""
@Time ： 2025/7/30 15:46
@Auth ： fengbangwei
@File ：evaluate.py

"""
import torch.cuda
import json
from torch.utils.data import DataLoader
from deepseek.week13.bert_tuning.common import generate_sentence


class ValidDataGenerator:
    def __init__(self, config):
        self.data = []
        self.path = config['valid_data_path']
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.data.append([content, title])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(config, shuffle=True):
    vdg = ValidDataGenerator(config)
    dl = DataLoader(vdg, batch_size=config["valid_batch_size"], shuffle=shuffle)
    return dl


class Evaluator:
    def __init__(self, config, model, tokenizer, logger):
        self.stats_dict = None
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config)
        self.tokenizer = tokenizer

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        # 重置统计计数器
        self.stats_dict = {"correct": 0, "wrong": 0}
        all_title_list, pred_title_list = [], []
        # 使用torch.no_grad上下文管理整个批次
        with torch.no_grad():
            # index 批次
            for index, batch_data in enumerate(self.valid_data):
                content, title = batch_data
                content_list = list(content) if not isinstance(content, list) else content
                title_list = list(title) if not isinstance(title, list) else title
                # 使用zip同时遍历content和title
                for single_content, single_title in zip(content_list, title_list):
                    pred_title = generate_sentence(single_content, self.model, self.tokenizer)
                    # print(pred_title)
                    all_title_list.append(single_title)
                    pred_title_list.append(pred_title)

        self.write_stats(all_title_list, pred_title_list)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        # assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            # print(true_label, pred_label)
            # 去除多余空格并清理输出
            input_text = pred_label.split("[SEP]")[0].strip()
            output_text = pred_label.split("[SEP]")[1].strip()
            # 去除中文字符间的空格
            input_text = "".join(input_text.split())
            output_text = "".join(output_text.split())
            # print(input_text)
            # print(output_text)
            print(input_text.replace('[CLS]', '') + ' -> ' + output_text)
            # 把 [UNK] 替换成空 用output_text 是否包含在true_label 包含则认为预测正确
            if output_text.replace('[UNK]', '') in true_label:
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)


if __name__ == '__main__':
    # vdg = ValidDataGenerator(Config)
    # print(vdg.__getitem__(0))
    print('as' in "asdklask")
