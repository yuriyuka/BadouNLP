# 改用交叉熵实现一个多分类任务,五维随机向量最大的数字在哪维就属于哪一类。

import numpy as np
import torch
from torch.onnx.symbolic_opset9 import tensor


class TorchModel(torch.nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        #torch.nn.CrossEntropyLoss() 是类 ，torch.nn.functional.cross_entropy()是函数 用哪个都可以
        self.loss = torch.nn.CrossEntropyLoss()  # loss函数采用交叉熵损失，交叉熵自带softmax
        self.layer1 = torch.nn.Linear(5, 5, 1)  # 五分类任务就是映射到5维，bias 1 就是有+b
        #最后不用再加一个激活函数，因为计算loss的函数已经包含了softmax

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)   # 为了和torch.nn.CrossEntropyLoss()的输入格式一致，输入是[1,5]的tensor
        y_pred = self.layer1(x)  # shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        return y_pred


# 计算5维向量的标签
def find_max(v):
    a = []
    vector = np.zeros(5)
    for x in range(5):
        if a == [] or v[x] > v[a[-1]]:
            a.append(x)
    return a[-1]

# 构建样本
def build_sample():
    vector_list = []
    for i in range(1000000):
        vector_list.append(torch.FloatTensor(np.random.rand(5)))
    return vector_list


# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_input = torch.FloatTensor([0.1465, 0.8398, 0.1783, 0.1029, 0.8920])
    test_target = 4

    with torch.no_grad():
        test_pred = model(test_input)
        print("模型预测。   ")
        print(test_pred)
        print(torch.argmax(test_pred).item())


def BGD():                        #全样本梯度下降法
    # 模型输入张量X
    train_data = build_sample()
    # 张量标签Y
    target = []

    for i in train_data:
        target.append(find_max(i))

    # 模型输出张量Y_pred
    model = TorchModel()
    model.train()
    watch_loss = []
    learning_rate = 0.00001
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  #目前用得多的优化器adam
    for i in range(1000000):
        Y_pred = model(train_data[i])
        Y = torch.tensor([target[i]], dtype=torch.long)
        loss = model.loss(Y_pred, Y)
        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        optim.zero_grad()  # 梯度归零
        watch_loss.append(loss.item())
        if i % 10000 == 0:
            print("=========\nloss:%f" % (np.mean(watch_loss)))
    evaluate(model)

def Mini_batch():
    model = TorchModel()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 尝试增大学习率
    counter = 0
    # 生成随机数据（5维向量，标签是最大值的索引）
    train_data = [torch.rand(5) for _ in range(100000)]
    target = [torch.argmax(torch.rand(5)).item() for _ in range(100000)]  # 真实标签
    batch_size = 100
    num_epochs = 1000
    for epoch in range(num_epochs):  # 训练100轮
        total_loss = 0

        for i in range(len(train_data)):
            x = train_data[i]
            y_true = torch.tensor([target[i]], dtype=torch.long)  # 标签必须是 LongTensor
            counter += 1
            if counter  == batch_size:
                y_pred = model(x)
                # 计算损失
                loss = model.loss(y_pred, y_true)
                total_loss += loss.item()
                # 反向传播和优化
                optim.zero_grad()
                loss.backward()
                optim.step()
                counter = 0

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_data)}")
    evaluate(model)
if __name__ == '__main__':
    BGD()   #全样本梯度下降法
    Mini_batch() #小批量梯度下降法
