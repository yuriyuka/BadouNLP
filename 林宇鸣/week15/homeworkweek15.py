
# coding=utf-8
import os
import re
from collections import defaultdict


def get_files_from_directory(directory):
    """读取指定目录下的所有.txt文件"""
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            files.append(os.path.join(directory, filename))
    return files


def read_corpus(file_paths):
    """读取文本文件并合并内容"""
    text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text += file.read() + "\n"
    return text


def get_stats(ids):
    """统计二元组出现次数"""
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids, pair, idx):
    """合并二元组"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def build_vocab(corpus, vocab_size):
    """构建词表"""
    # 将文本编码为字节值
    tokens = list(corpus.encode("utf-8"))
    merges = {}
    idx = 256  # 从256开始，因为前256是原始字节

    while len(merges) < vocab_size - 256:
        stats = get_stats(tokens)
        if not stats:
            break  # 没有更多的二元组可以合并
        pair = max(stats, key=stats.get)  # 找到出现次数最多的二元组
        merges[pair] = idx
        tokens = merge(tokens, pair, idx)
        idx += 1

    return merges


def save_vocab(merges, directory):
    """将词表保存到文件"""
    with open(os.path.join(directory, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for pair, idx in merges.items():
            f.write(f"{pair[0]} {pair[1]} {idx}\n")


def decode(ids, vocab):
    """解码函数"""
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text, merges):
    """编码函数"""
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # 没有可合并的对
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


def main():
    directory = r"D:\codeLearning\code\badouweek15\homeworkweek15\Heroes"
    file_paths = get_files_from_directory(directory)
    corpus = read_corpus(file_paths)

    vocab_size = 10000
    merges = build_vocab(corpus, vocab_size)

    # 保存词表
    save_vocab(merges, directory)

    # 准备字典以便解码
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    # 测试编码和解码
    test_string1 = "我爱玩英雄联盟"
    encoded1 = encode(test_string1, merges)
    print("编码结果1:", encoded1)

    test_string2 = "英雄联盟里面一共有多少个英雄"
    encoded2 = encode(test_string2, merges)
    print("编码结果2:", encoded2)

    # 解码
    decoded1 = decode(encoded1, vocab)
    decoded2 = decode(encoded2, vocab)

    print("解码结果1:", decoded1)
    print("解码结果2:", decoded2)


if __name__ == "__main__":
    main()

# 编码结果1: [1078, 3363, 2735, 397, 7015, 2369]
# 编码结果2: [397, 7015, 2369, 1884, 280, 2811, 5082, 983, 323, 397]
# 解码结果1: 我爱玩英雄联盟
# 解码结果2: 英雄联盟里面一共有多少个英雄