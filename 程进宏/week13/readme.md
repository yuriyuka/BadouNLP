## lora 处理 ner
### main.py 
1. 定义微调策略 LoRa
2. 创建模型保存目录(调用 os)
3. 加载训练数据集(调用 load_data 函数)
4. 加载模型并传入微调策略
5. 标识是否使用 gpu
6. 加载优化器
7. 加载效果测试类
8. 训练主流程（循环函数）
   1. epoch 循环
      1. batch 循环
         1. 梯度归零
         2. cuda
         3. 读取批次数据
         4. 根据输入和标签传入模型计算 loss
         5. 反向传播：loss.backward()
         6. optimizer.step()
         7. 保存 loss
      2. 模型评估
9. 保存模型
### loader.py
1. DataGenerator 类
   1. 构造函数
      1. 传入 data_path、config；参数有 config、path、tokenizer、sentences、schema、load()函数调用
   2. load 函数
      1. 创建一个空的 data 列表，读取训练数据（注意txt格式和json格式的数据读取方式，以及如何进行构造）
      2. 遍历训练数据文件，\n\n划分每一句话，\n划分每一个字符样本
      3. 构造格式：（需要进行 padding）
         1. labels=[[8, label_1, label_2...],[8, label_a, label_b...]] （其中8为 cls_token ）
         2. input_ids = self.encode_sentence(sentenece)  （其中 sentenece 为句子）
   3. 其他函数
2. load_data 函数
3. load_vocab 函数
### model.py
1. ConfigWrapper 类
   1. 构造函数
      1. 传入 config
   2. to_dict 函数
      1. 返回 config
2. TorchModel 类
   1. 构造函数
      1. 传入 config，参数有 config、max_length、class_num、bert、classify、crf_layer、use_crf、loss
   2. forward 函数
      1. 传入 x,target=none; （target=none不计算loss）
      2. x 过 bert 层，再过 classify 层
      3. 判断 target 和 use_crf；然后决定是否计算 crf loss 还是直接返回预测结果
3. choose_optimizer 函数
### evaluate.py
1. Evaluate 类
### config.py
1. 配置参数(字典)
### predict.py
1. 模型效果测试
   1. 构造函数
   2. load_schema 函数
   3. load_vocab 函数
   4. encoder_sentence 函数
   5. decode 函数
   6. predict 函数
