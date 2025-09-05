#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import List, Dict, Tuple


class BPETokenizer:
    
    def __init__(self, vocab_size: int = 276):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        # 统计词对频率
        counts = {}
        for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        # 合并指定的词对
        newids = []
        i = 0
        while i < len(ids):
            # if we are not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text: str):
        # 将文本转换为UTF-8字节序列
        tokens = text.encode("utf-8")  # raw bytes
        tokens = list(map(int, tokens))  # convert to a list of integers in range 0..255
        
        print(f"原始文本长度: {len(text)}")
        print(f"字节序列长度: {len(tokens)}")
        
        # 计算需要合并的次数
        num_merges = self.vocab_size - 256
        ids = list(tokens)
        
        # 迭代合并最频繁的词对
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
                
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"合并 {pair} 为新的token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
        
        # 构建词表
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
        print(f"训练完成！词表大小: {len(self.vocab)}, 合并规则数: {len(self.merges)}")
        print(f"压缩比: {len(tokens) / len(ids):.2f}X")
    
    def encode(self, text: str) -> List[int]:
        # 将文本转换为UTF-8字节序列
        tokens = list(text.encode("utf-8"))
        
        # 应用所有合并规则
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        # 将token ID转换为字节序列
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # 解码为UTF-8文本
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def save_model(self, filepath: str):
        # 保存模型
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()},
            'vocab': {str(idx): list(self.vocab[idx]) for idx in self.vocab}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_model(self, filepath: str):
        # 加载模型
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        self.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
    
    
def main():
    base_text = """A Programmer's Introduction to Unicode March 3, 2017 · Coding · 22 Comments  Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺\u200c🇳\u200c🇮\u200c🇨\u200c🇴\u200c🇩\u200c🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to "support Unicode" in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception."""
    bpe_tokenizer = BPETokenizer(vocab_size=276)
    bpe_tokenizer.train(base_text)
    
    small_tests = [
        "A Programmer's Introduction to Unicode",
        "안녕하세요 👋 (hello in Korean!)",
        "BPE是一种常用的分词方法"
    ]
    
    for tt in small_tests:
        print(f"\n测试文本: {tt}")
        ids = bpe_tokenizer.encode(tt)
        print(f"编码结果: {ids}")
        print(f"Token数量: {len(ids)}")
        dec = bpe_tokenizer.decode(ids)
        print(f"解码结果: {dec}")
        print(f"编码解码一致性: {tt == dec}")

if __name__ == "__main__":
    main()
