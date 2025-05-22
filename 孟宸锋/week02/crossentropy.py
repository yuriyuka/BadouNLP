
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 第一步，X:随机生成10000个五维数据，y:然后得到其中最大的那个数据
def build_sample():
    x = np.random.randint(1, 101, size=5)
    y = np.argmax(x)  # 返回最大值的位置索引，如 3
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)



# 第二步，用pytorch 建立一个模型
# 1. 定义模型
class TorchModel(torch.nn.Module):
    def __init__(self, input_size,):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 32)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(32, input_size)

        #self.cross_entropy = torch.nn.CrossEntropyLoss() # loss 函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        x = self.activation1(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        x = self.linear2(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        x = self.activation2(x)  # (batch_size, hidden_size) -> (batch_size, hidden_size)
        y_pred = self.linear3(x)  # (batch_size, hidden_size) -> (batch_size, input_size)

        return y_pred  # 输出预测结果


 
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            y_pred = model(inputs)
            _, predicted = torch.max(y_pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
    print('Accuracy of the model on the test set: %d %%' % (accuracy))
    return accuracy



def main():
    epoch = 1000
    batch_size = 32
    learning_rate = 0.001
    input_size = 5
    train_sample_per_epoch = 500
    k_fold = 10
    #建立模型
    model = TorchModel(input_size)
    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()
    log = []

    # 1. 生成数据集
    X, Y = build_dataset(10000)

    # 2. 将数据集分为训练集和测试集
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size

    train_dataset = torch.utils.data.TensorDataset(X[:train_size], Y[:train_size])
    test_dataset = torch.utils.data.TensorDataset(X[train_size:], Y[train_size:])

    # 3. 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 训练模型
    for epoch in range(epoch):
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            logits = model(inputs)  # 模型预测

            labels = labels.squeeze().long()
            loss = loss_fn(logits, labels)  # 计算损失
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

        if (epoch + 1) % 100 == 0:
            print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epoch, loss.item()))
            acc = evaluate(model, test_loader)  # 测试本轮模型结果
            log.append([acc, loss.item()])
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
    


if __name__ == "__main__":
    main()
