# 用bpe完成词表构建和文本的序列化

"""
    思路：
    正确BPE流程：
    1. 初始化词表为所有基础字符
    2. 统计所有相邻符号对的频率
    3. 合并最高频的符号对，生成新token
    4. 更新语料（替换所有该符号对）
    5. 重复2-4直到达到目标词表大小
    6. 输出最终词表
"""
import os
from collections import defaultdict


def get_stats(ids: list[int]):
    counts = defaultdict(int)
    for i in range(len(ids) - 1):
        counts[ids[i], ids[i + 1]] += 1
    return counts


def merge_pair(pair, ids, idx):
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


class BPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merge = {}

    def train(self, text: str):
        assert self.vocab_size > 256
        num_merges = self.vocab_size - 256
        ids = list(text.encode("utf-8"))
        for i in range(num_merges):
            counts = get_stats(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            idx = 256 + i
            ids = merge_pair(pair, ids, idx)
            self.merge[pair] = idx
            print(f"merge pair:{pair} to idx:{idx}")
        self.vocab = {i: chr(i) for i in range(256)}
        for k, v in self.merge.items():
            self.vocab[v] = k

    def encode(self, text: str):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = get_stats(ids)
            pair = max(counts, key=counts.get)
            if pair not in self.merge:
                break
            idx = self.merge[pair]
            ids = merge_pair(pair, ids, idx)
        return ids

    def decode(self, ids: list[int]):
        while True:
            new_ids = []
            flag = False
            for i in range(len(ids)):
                if ids[i] >= 256:
                    pair = self.vocab[ids[i]]
                    new_ids.extend(pair)
                    flag = True
                else:
                    new_ids.append(ids[i])
            if not flag:
                break
            ids = new_ids
        return bytes(ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=10000)
    all_text = ""
    for file_name in os.listdir("Heroes"):
        if file_name.endswith(".txt"):
            with open(os.path.join("Heroes", file_name), "r", encoding="utf-8") as f:
                text = f.read()
                all_text += text
    tokenizer.train(all_text)
    encoded = tokenizer.encode("This is the Hugging Face Course...")
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
