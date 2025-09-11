#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BPE (Byte Pair Encoding) 算法实现
用于词表构建和文本序列化
"""
import re
from collections import defaultdict
import json

class BPE:
    def __init__(self):
        # 初始化词表和合并规则
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.reverse_special_tokens = {}
    
    def _get_vocab(self, corpus):
        """从语料库中统计单词频率"""
        vocab = defaultdict(int)
        for line in corpus:
            words = line.strip().split()
            for word in words:
                # 在词尾添加结束标记</w>
                vocab[word + '</w>'] += 1
        return vocab
    
    def _get_pairs(self, word):
        """获取单词中的字符对"""
        symbols = word.split()
        # 如果单词只有一个字符，返回空字典
        if len(symbols) <= 1:
            return {}
        
        pairs = defaultdict(int)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    def _merge_vocab(self, pair, word_str):
        """合并单词中的字符对"""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        return p.sub(''.join(pair), word_str)
    
    def build_vocab(self, corpus, vocab_size, special_tokens=None):
        """构建BPE词表"""
        # 初始化特殊标记
        if special_tokens:
            self.special_tokens = special_tokens
            self.reverse_special_tokens = {v: k for k, v in special_tokens.items()}
        
        # 统计词汇频率
        vocab = self._get_vocab(corpus)
        
        # 初始化词汇表为所有字符
        chars = set()
        for word in vocab:
            chars.update(word.replace(' ', ''))
        
        # 添加特殊标记到词汇表
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
        
        # 设置初始词汇表索引
        next_idx = len(self.special_tokens) if self.special_tokens else 0
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = next_idx
                next_idx += 1
        
        # 执行BPE合并操作
        current_vocab = {word: word.replace('', ' ').strip() for word in vocab}
        
        # 词汇表大小应大于初始字符集大小
        assert vocab_size > len(self.vocab), "词汇表大小应大于初始字符集大小"
        
        # 执行合并直到达到目标词汇表大小
        while len(self.vocab) < vocab_size:
            # 统计所有字符对
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                word_pairs = self._get_pairs(current_vocab[word])
                for pair, p_freq in word_pairs.items():
                    pairs[pair] += p_freq * freq
            
            # 如果没有更多字符对可以合并，结束循环
            if not pairs:
                break
            
            # 找到出现频率最高的字符对
            best = max(pairs, key=pairs.get)
            
            # 合并字符对到词汇表
            merged_token = ''.join(best)
            self.vocab[merged_token] = next_idx
            next_idx += 1
            
            # 记录合并规则
            self.merges[best] = merged_token
            
            # 更新词汇表中的单词
            current_vocab = {word: self._merge_vocab(best, current_vocab[word]) for word in current_vocab}
        
        return self.vocab
    
    def tokenize(self, text):
        """将文本标记化"""
        # 预处理文本，添加结束标记</w>
        words = text.strip().split()
        words_with_end = [word + '</w>' for word in words]
        
        # 对每个单词应用BPE合并规则
        tokenized = []
        for word in words_with_end:
            # 初始化为字符级别
            tokens = list(word)
            
            # 应用合并规则
            while True:
                # 检查是否有可以合并的字符对
                pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
                merge_candidates = {pair: i for i, pair in enumerate(pairs) if pair in self.merges}
                
                # 如果没有可合并的字符对，结束循环
                if not merge_candidates:
                    break
                
                # 找到最早出现的可合并字符对（按照合并顺序）
                # 注意：这里简化处理，实际应按照合并顺序优先级
                pair_to_merge = min(merge_candidates.keys(), key=lambda p: list(self.merges.keys()).index(p))
                idx = merge_candidates[pair_to_merge]
                
                # 合并字符对
                tokens = tokens[:idx] + [self.merges[pair_to_merge]] + tokens[idx+2:]
            
            # 将标记添加到结果中
            tokenized.extend(tokens)
        
        return tokenized
    
    def encode(self, text):
        """将文本编码为索引序列"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        tokens = []
        for idx in indices:
            # 查找索引对应的标记
            token = next((t for t, i in self.vocab.items() if i == idx), '<unk>')
            tokens.append(token)
        
        # 合并标记并替换结束标记</w>为空格
        text = ''.join(tokens).replace('</w>', ' ')
        # 移除多余的空格
        return re.sub(r'\s+', ' ', text).strip()
    
    def save_vocab(self, filepath):
        """保存词汇表到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath):
        """从文件加载词汇表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

# 示例用法
if __name__ == "__main__":
    # 示例语料库
    corpus = [
        "hello world",
        "hello there",
        "world is beautiful",
        "hello beautiful world"
    ]
    
    # 创建BPE实例
    bpe = BPE()
    
    # 定义特殊标记
    special_tokens = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    
    # 构建词表
    vocab_size = 30
    vocab = bpe.build_vocab(corpus, vocab_size, special_tokens)
    
    print("构建的词表:")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"{token}: {idx}")
    
    # 测试编码和解码
    test_text = "hello beautiful"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)
    
    print(f"\n测试文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    
    # 保存词表
    bpe.save_vocab('bpe_vocab.json')
    print("\n词表已保存到bpe_vocab.json")
