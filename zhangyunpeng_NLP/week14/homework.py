import json
import os


def get_stats(ids):
    """统计相邻token对的频率"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """将ids中所有pair替换为idx"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPE_Tokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def train(self, folder_path):
        """训练BPE模型"""
        # 读取所有文本数据
        texts = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    texts.append(file.read())

        # 将所有文本转换为utf-8编码的字节序列
        tokens = []
        for text in texts:
            tokens.extend(list(text.encode("utf-8")))

        print(f"训练数据总字节数: {len(tokens)}")

        # 执行BPE训练
        num_merges = self.vocab_size - 256
        ids = list(tokens)  # 复制以便不破坏原始列表

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx

            if i % 100 == 0:
                print(f"合并进度: {i}/{num_merges}")

        # 构建词汇表
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        print(f"训练完成，最终词汇表大小: {len(self.vocab)}")

    def encode(self, text):
        """将文本编码为token序列"""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # 没有更多可以合并的
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        """将token序列解码为文本"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def save_model(self, path):
        """保存模型到文件"""
        model_data = {
            "vocab_size": self.vocab_size,
            "merges": [(list(k), v) for k, v in self.merges.items()],
            "vocab": {k: list(v) for k, v in self.vocab.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load_model(self, path):
        """从文件加载模型"""
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        self.vocab_size = model_data["vocab_size"]
        self.merges = {tuple(k): v for k, v in model_data["merges"]}
        self.vocab = {int(k): bytes(v) for k, v in model_data["vocab"].items()}


def main():
    # 创建并训练BPE分词器
    tokenizer = BPE_Tokenizer(vocab_size=1024)
    print("开始训练BPE模型...")
    tokenizer.train("prts")

    # 保存模型
    tokenizer.save_model("bpe_model.json")
    print("模型已保存到 bpe_model.json")

    # 测试编码和解码
    test_text = "流明是伊比利亚出身的阿戈尔人"
    print(f"\n测试文本: {test_text}")

    # 编码
    encoded = tokenizer.encode(test_text)
    print(f"编码结果: {encoded}")

    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")

    # 验证编码解码是否正确
    print(f"编码解码是否正确: {test_text == decoded}")

    # 测试文件中的文本
    with open("prts/流明.txt", "r", encoding="utf-8") as f:
        liming_text = f.read()

    print(f"\n长文本测试 (流明.txt 前100字):")
    test_long_text = liming_text[:100]
    print(f"原文: {test_long_text}")

    encoded_long = tokenizer.encode(test_long_text)
    print(f"编码长度: {len(encoded_long)} (原文UTF-8字节长度: {len(test_long_text.encode('utf-8'))})")

    decoded_long = tokenizer.decode(encoded_long)
    print(f"解码是否一致: {test_long_text == decoded_long}")


if __name__ == "__main__":
    main()
