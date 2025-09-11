# -*- coding: utf-8 -*-

"""
配置参数信息
可以在 main.py 文件中修改这些参数来训练模型观察准确率等
如：修改 model_type、learning_rate、hidden_size、batch_size、pooling_style 等
"""
Config = {
    "model_path": "output",                 # 模型保存路径
    "train_data_path": "..\\work\\train_data.txt",    # 训练数据路径
    "valid_data_path": "..\\work\\valid_data.txt",    # 验证数据路径
    "vocab_path": "..\\work\\chars.txt",              # 词表路径
    "pretrain_model_path": r"D:\worksapce\ai_workspace\nlp20\week6\bert-base-chinese",# bert 预训练模型路径
    "model_type": "bert",                   # 模型类型
    "max_length": 30,                       # 句子最大长度
    "hidden_size": 256,                     # 隐藏层维度，bert 隐藏层维度固定为768
    "num_layers": 2,                        # 神经网络层数: 控制RNN/LSTM/GRU的层数或堆叠层数
    "kernel_size": 3,                       # CNN卷积核大小
    "epoch": 5,                             # 训练轮数
    "batch_size": 16,                       # 批次大小
    "optimizer": "adam",                    # 优化器
    "learning_rate": 1e-5,                  # 学习率
    "pooling_style": "avg",                 # 池化方式: avg\max
    "seed": 987                             # 随机种子：确保每次运行结果一致
    # "class_num": 2      分类数，在 loader.py 中定义
}

