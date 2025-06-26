from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''


if __name__ == '__main__':
    # 自制
    bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    keys = state_dict.keys()
    print(keys)  # 查看所有的权值矩阵名称
    print("参数量有 " + str(keys.__len__()) + " 个")  # 计算权值矩阵的数量
