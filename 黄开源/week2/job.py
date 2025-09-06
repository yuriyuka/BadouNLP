# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


# 构造模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 定义要使用哪些网络层
        self.linear = nn.Linear(input_size, 5)  # 定义一个线性层，输出维度是5
        # 多分类任务一般不在模型中添加激活函数，而是在损失函数中处理
        # 定义损失函数
        self.loss = nn.functional.cross_entropy  # 交叉熵损失函数

    def forward(self, x, y=None):
        # 定义这些网络层是如何组合的
        x = self.linear(x)
        # y_pred = self.activation(x)
        # 交叉熵损失韩式内部回自动应用softmax，不需要手动添加激活函数
        if y is not None:
            return self.loss(x, y.long())  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
        # 转换为numpy数组后再转换为tensor
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # print("训练集：", train_x, train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 每次取出20个样本
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算loss
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    # 打印权重
    print("训练好的模型权重：" + str(model.state_dict()))

    # 开启测试模式
    model.eval()
    # 不计算梯度
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        # 对于多分类，需要找出概率最大的类别
        probabilities = torch.softmax(result, dim=1)  # 应用softmax获取概率
        predicted_classes = torch.argmax(probabilities, dim=1)  # 获取预测类别

    for i, (vec, prob, pred_class) in enumerate(zip(input_vec, probabilities, predicted_classes)):
        print(f"输入：{vec}")
        print(f"各类别概率：{prob}")
        print(f"预测类别：{pred_class}")
        print(f"实际最大值位置：{np.argmax(vec)}")
        print("-" * 50)

# 启动入口
if __name__ == "__main__":
    # 模型训练过程
    # main()
    # 模型预测过程
    X = [0.01 * x for x in range(100)]
    test_vec = [np.random.random(5).tolist() for _ in range(10)]
    # for item in test_vec:
    #     print(item)
    predict("model.bin", test_vec)
