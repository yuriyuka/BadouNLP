import json
from collections import Counter
from typing import List, Dict, Tuple


def build_bpe_vocab(corpus_file: str, num_merges: int = 100) -> Tuple[Dict[bytes, int], List[Tuple[bytes, bytes]]]:
    """构建BPE词表"""
    vocab = {bytes([i]): i for i in range(256)}
    merges = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    tokenized_corpus = []
    for line in corpus.split('\n'):
        if line.strip():
            tokens = [bytes([b]) for b in line.encode('utf-8')]
            tokenized_corpus.append(tokens)

    for _ in range(num_merges):
        pair_freq = Counter()
        for tokens in tokenized_corpus:
            for i in range(len(tokens) - 1):
                pair_freq[(tokens[i], tokens[i + 1])] += 1

        if not pair_freq:
            break

        best_pair = max(pair_freq, key=pair_freq.get)
        new_token = best_pair[0] + best_pair[1] # 创建新词
        vocab[new_token] = len(vocab)
        merges.append(best_pair)

        for idx, tokens in enumerate(tokenized_corpus):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]: # 找到匹配对更新词表
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokenized_corpus[idx] = new_tokens

    return vocab, merges


def serialize_bpe(vocab: Dict[bytes, int], merges: List[Tuple[bytes, bytes]], filepath: str):
    """序列化BPE模型"""
    data = {
        'vocab': {k.hex(): v for k, v in vocab.items()},
        'merges': [(p[0].hex(), p[1].hex()) for p in merges]
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def deserialize_bpe(filepath: str) -> Tuple[Dict[bytes, int], List[Tuple[bytes, bytes]]]:
    """反序列化BPE模型"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vocab = {bytes.fromhex(k): v for k, v in data['vocab'].items()}
    merges = [(bytes.fromhex(p[0]), bytes.fromhex(p[1])) for p in data['merges']]

    return vocab, merges


def encode(text: str, vocab: Dict[bytes, int], merges: List[Tuple[bytes, bytes]]) -> List[int]:
    tokens = [bytes([b]) for b in text.encode('utf-8')]

    for pair in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return [vocab[token] for token in tokens]


def decode(ids: List[int], vocab: Dict[bytes, int]) -> str:
    id_to_token = {v: k for k, v in vocab.items()}
    tokens = [id_to_token[id] for id in ids]
    return b''.join(tokens).decode('utf-8', errors='replace')


# 示例代码
if __name__ == "__main__":
    # 构建BPE词表
    vocab, merges = build_bpe_vocab("corpus.txt", num_merges=50)
    print(f"词表大小: {len(vocab)}, 合并规则: {len(merges)}")

    # 测试编码解码
    test_text = "你好世界"
    ids = encode(test_text, vocab, merges)
    decoded = decode(ids, vocab)
    print(f"原文: {test_text}")
    print(f"编码: {ids}")
    print(f"解码: {decoded}")