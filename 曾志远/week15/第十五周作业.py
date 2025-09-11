# bpe压缩算法
import json
import os

# 对应表
merges = {}
vocab = {}


# 读取文件
def open_file(path: str) -> str:
    res = ""
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            res += line.strip()
    return res


# 将文本文档转为UTF-8编码
def translate_str_to_UTF8Code(word: str) -> list:
    tokens = word.encode('utf8')
    tokens = list(map(int, tokens))
    return tokens


# 获取两两组合次数最多的一对组合
def get_frequency_top(ids: list):
    dict_f = {}
    for pair in zip(ids, ids[1:]):
        num = dict_f.get(pair, 0) + 1
        dict_f[pair] = num
    return sorted(dict_f.items(), key=lambda x: x[1], reverse=True)[0][0]


# 将文档压缩
def merge(tokens: list, vocab_size: int) -> list:
    print("-------------------------------------------------------")
    print(f"原始列表长度: {len(tokens)}")
    print(f"tokens: {tokens}")
    print("-------------------------------------------------------")
    temp_tokens = []
    merged_tokens = tokens
    num_merge = vocab_size - 255
    for i in range(num_merge):
        # 获取两两组合最多的组合
        max_pair = get_frequency_top(merged_tokens)
        j = 0
        while j < len(merged_tokens):
            if j + 1 < len(merged_tokens) and merged_tokens[j] == max_pair[0] and merged_tokens[j + 1] == max_pair[1]:
                token = 255 + i + 1
                merges[max_pair] = token
                temp_tokens.append(token)
                j = j + 2
            else:
                token = merged_tokens[j]
                temp_tokens.append(token)
                j = j + 1
        merged_tokens = temp_tokens
        temp_tokens = []
    print("-------------------------------------------------------")
    print(f"合并后列表长度: {len(merged_tokens)}")
    print(f"merged_tokens: {merged_tokens}")
    print("-------------------------------------------------------")
    return merged_tokens


# 解码
def decode(tokens):
    tokens = b"".join(vocab[idx] for idx in tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text


# 编码
def encode(tokens: list, vocab_size: int):
    merged_tokens = merge(tokens, vocab_size)
    return merged_tokens


# 构建词表
def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return merges, vocab


# 写入文件
def write_into_file(merges, vocab):
    if not os.path.isdir("model_output"):
        os.mkdir("model_output")
    vocab_path = os.path.join("model_output", "vocab.json")
    merges_path = os.path.join("model_output", "merges.json")
    vocab_new_data = {k: v.hex() for k, v in vocab.items()}
    merges_new_data = {str(key): value for key, value in merges.items()}
    with open(merges_path, 'w') as f:
        json.dump(merges_new_data, f)
    with open(vocab_path, 'w') as f:
        json.dump(vocab_new_data, f)


if __name__ == '__main__':
    corpus = open_file('corpus.txt')
    # s = "警方迪斯科飞机上的纠纷看见撒旦发的老师都放假撒JFK就圣诞快乐房价开始垃圾分类看"
    utf_8_list = translate_str_to_UTF8Code(corpus)
    merged_tokens = encode(utf_8_list, 500)
    # s2 = decode(merged_tokens)
    merges, vocab = build_vocab(merges)
    write_into_file(merges, vocab)
