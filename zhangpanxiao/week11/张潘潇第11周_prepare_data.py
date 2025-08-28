import json
from datasets import Dataset

#使用新闻数据尝试实现sft训练。 数据预处理脚本
def convert_format(sample):
    """将新闻数据转换为指令格式"""
    return {
        "instruction": "请根据标题生成新闻正文",
        "input": sample["title"],
        "output": sample["content"]
    }


def process_data(input_file, output_file):
    # 加载原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 转换格式
    formatted_data = [convert_format(d) for d in data]

    # 创建Dataset并保存
    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk(output_file)
    print(f"保存处理后的数据到 {output_file}, 样本数: {len(dataset)}")


if __name__ == "__main__":
    process_data(
        input_file="data/raw_news.json",
        output_file="data/processed/sft_dataset"
    )
