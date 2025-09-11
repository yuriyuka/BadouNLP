# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from collections import defaultdict

# --- BPE Helper Functions ---

def get_stats(ids):
    """
    计算ids中相邻词对的频率
    """
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

def merge(ids, pair, idx):
    """
    将ids中的所有指定词对替换为新的idx
    """
    newids = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i+1]) == pair:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# --- BPE Training Function ---

def train_bpe(text_data, vocab_size):
    """
    训练BPE分词器
    text_data: 训练语料，字符串
    vocab_size: 目标词表大小
    """
    # 初始词表：所有可能的字节 (0-255)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    
    # 存储合并规则
    merges = {}
    
    # 将文本转换为初始字节序列
    # 对于多行文本，将其连接起来，然后编码
    byte_tokens = list(text_data.encode("utf-8"))
    
    # 进行合并操作
    for i in range(256, vocab_size):
        # 计算当前token序列中所有词对的频率
        stats = get_stats(byte_tokens)
        
        if not stats:
            break # 没有可以合并的词对
            
        # 找到频率最高的词对
        best_pair = max(stats, key=stats.get)
        
        # 将该词对添加到合并规则中，并分配新的token ID
        merges[best_pair] = i
        vocab[i] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 将所有出现该词对的地方进行合并
        byte_tokens = merge(byte_tokens, best_pair, i)
        
        print(f"合并 {best_pair} -> {i} ({vocab[i].decode('utf-8', errors='ignore')}), 词表大小: {len(vocab)}")
    
    return merges, vocab

# --- Encode and Decode Functions ---

def encode(text, merges, vocab):
    """
    给定一个字符串，使用BPE规则将其编码为token ID列表
    """
    # 将文本转换为初始字节序列
    tokens = list(text.encode("utf-8"))
    
    # 迭代应用合并规则，直到没有更多可以合并的词对
    while True:
        stats = get_stats(tokens)
        
        # 找到当前token序列中最优（在merges中）的词对
        # key=lambda p: merges.get(p, float("inf")) 表示如果pair不在merges中，则优先级最低
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        
        # 如果最优先的词对不在merges中，说明没有更多可合并的了
        if pair not in merges:
            break
        
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids, vocab):
    """
    给定token ID列表，将其解码回Python字符串
    """
    # 将token ID序列转换回字节序列
    tokens = b"".join(vocab[idx] for idx in ids)
    # 将字节序列解码为UTF-8字符串
    text = tokens.decode("utf-8", errors="replace")
    return text

# --- Main Execution ---

if __name__ == "__main__":
    # --- 准备训练数据 ---
    # 使用较长的文本来训练BPE，以获得更有意义的合并
    # 这里使用您之前提供的valtext作为训练数据
    text = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
    
    # 增加一些包含特殊字符和韩语的文本，确保BPE能处理多语言
    text_data = text + "안녕하세요 👋 (hello in Korean!) " + \
                "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

    # --- 训练BPE分词器 ---
    print("--- 开始训练BPE分词器 ---")
    # 目标词表大小，例如 500 个 token (256个字节 + 244个合并)
    BPE_VOCAB_SIZE = 500 
    merges, vocab = train_bpe(text_data, BPE_VOCAB_SIZE)
    print("--- BPE分词器训练完成 ---")
    print(f"最终词表大小: {len(vocab)}")
    print(f"合并规则数量: {len(merges)}")
    print("\n部分合并规则:")
    for (pair, new_id) in list(merges.items())[:10]:
        print(f"  {pair} -> {new_id} ({vocab[new_id].decode('utf-8', errors='ignore')})")


    # --- 测试 encode 和 decode 功能 ---
    print("\n--- 测试编码和解码 ---")

    test_string_1 = "A Programmer’s Introduction to Unicode"
    encoded_ids_1 = encode(test_string_1, merges, vocab)
    decoded_text_1 = decode(encoded_ids_1, vocab)
    print(f"原始文本: '{test_string_1}'")
    print(f"编码IDs: {encoded_ids_1}")
    print(f"解码文本: '{decoded_text_1}'")
    print(f"解码是否匹配原始文本: {decoded_text_1 == test_string_1}")
    
    print("-" * 30)

    test_string_2 = "안녕하세요 👋 (hello in Korean!)"
    encoded_ids_2 = encode(test_string_2, merges, vocab)
    decoded_text_2 = decode(encoded_ids_2, vocab)
    print(f"原始文本: '{test_string_2}'")
    print(f"编码IDs: {encoded_ids_2}")
    print(f"解码文本: '{decoded_text_2}'")
    print(f"解码是否匹配原始文本: {decoded_text_2 == test_string_2}")

    print("-" * 30)

    valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system. Unicode encodes thousands of emoji, with the continued development thereof conducted by the Consortium as a part of the standard.[4] Moreover, the widespread adoption of Unicode was in large part responsible for the initial popularization of emoji outside of Japan. Unicode is ultimately capable of encoding more than 1.1 million characters."
    valtext2 = decode(encode(valtext, merges, vocab), vocab)
    print(f"长文本解码是否匹配原始文本: {valtext2 == valtext}")

    print("-" * 30)
    # 测试一些未在训练语料中但BPE应该能处理的文本
    test_string_3 = "这是一个新的中文句子。"
    encoded_ids_3 = encode(test_string_3, merges, vocab)
    decoded_text_3 = decode(encoded_ids_3, vocab)
    print(f"原始文本: '{test_string_3}'")
    print(f"编码IDs: {encoded_ids_3}")
    print(f"解码文本: '{decoded_text_3}'")
    print(f"解码是否匹配原始文本: {decoded_text_3 == test_string_3}")

    print("\n--- 编码原始文本并计算压缩率 ---")
    original_bytes_len = len(text_data.encode("utf-8"))
    encoded_tokens = encode(text_data, merges, vocab)
    encoded_ids_len = len(encoded_tokens)
    print(f"原始字节长度: {original_bytes_len}")
    print(f"编码Token ID长度: {encoded_ids_len}")
    print(f"压缩率 (字节/Token ID): {original_bytes_len / encoded_ids_len:.2f}X")
