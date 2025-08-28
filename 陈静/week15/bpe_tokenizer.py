# -*- coding: utf-8 -*-
"""
这个文件实现了一个完整的BPE分词系统，包括词汇表构建、文本编码解码和模型序列化
"""

import re  
import json  
import pickle  # 序列化模块，用于保存二进制数据
from collections import defaultdict, Counter  # 特殊字典类型，用于统计
from typing import List, Dict, Tuple, Optional, Set  # 类型提示，让代码更清晰


class BPEVocabBuilder:
  
    def __init__(self, vocab_size: int = 1000, min_frequency: int = 2):
        """
        vocab_size: 目标词汇表大小（默认1000个词）
        min_frequency: 字符对合并的最小频率阈值（默认2次）
        
        """
        self.vocab_size = vocab_size  # 保存目标词汇表大小
        self.min_frequency = min_frequency  # 保存最小频率阈值
        self.vocab = {}  # 最终的词汇表，格式：{token: id}
        self.merges = []  # 合并规则列表，记录每次合并的字符对
        
    def get_word_tokens(self, text: str) -> Dict[str, int]:
     
        # 使用正则表达式分割文本：\w+匹配词，[^\w\s]匹配标点符号
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        # 统计每个词的出现频率
        word_freqs = Counter(words)
        
        word_tokens = {}
        # 将每个词转换为字符序列，并添加结束标记</w>
        for word, freq in word_freqs.items():
            # 将词拆分为字符，用空格连接，最后加上结束标记
            # 例如："你好" -> "你 好 </w>"
            word_tokens[' '.join(word) + ' </w>'] = freq
            
        return word_tokens
    
    def get_pairs(self, word_tokens: Dict[str, int]) -> Dict[Tuple[str, str], int]:
       
        pairs = defaultdict(int)  # 使用defaultdict自动初始化为0
        
        # 遍历所有词token
        for word, freq in word_tokens.items():
            symbols = word.split()  # 将词token分割成字符列表
            # 统计相邻字符对
            for i in range(len(symbols) - 1):
                # 将相邻的两个字符作为一对，累加其出现频率
                pairs[(symbols[i], symbols[i + 1])] += freq
                
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], word_tokens: Dict[str, int]) -> Dict[str, int]:
        
        # 创建正则表达式模式，用于匹配要合并的字符对
        bigram = re.escape(' '.join(pair))  # 转义特殊字符
        # (?<!\S)表示前面不能是非空白字符，(?!\S)表示后面不能是非空白字符
        # 这确保我们只匹配完整的字符对
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        new_word_tokens = {}
        # 遍历所有词，将匹配的字符对合并
        for word in word_tokens:
            # 用合并后的字符替换原来的字符对
            new_word = p.sub(''.join(pair), word)
            new_word_tokens[new_word] = word_tokens[word]
            
        return new_word_tokens
    
    def build_vocab(self, texts: List[str]):
       
        print("开始构建BPE词汇表...")
        
        # 第一步：合并所有训练文本
        all_text = ' '.join(texts)
        print(f"训练文本总长度：{len(all_text)}个字符")
        
        # 第二步：获取初始词token表示
        word_tokens = self.get_word_tokens(all_text)
        print(f"初始词数量：{len(word_tokens)}")
        
        # 第三步：收集所有初始字符作为基础词汇
        vocab = set()
        for word in word_tokens.keys():
            for symbol in word.split():
                vocab.add(symbol)
        
        print(f"初始字符数量：{len(vocab)}")
        
        # 第四步：计算需要进行的合并次数
        num_merges = self.vocab_size - len(vocab)
        print(f"需要进行 {num_merges} 次合并操作")
        
        # 第五步：开始BPE迭代合并过程
        for i in range(num_merges):
            # 统计当前所有字符对的频率
            pairs = self.get_pairs(word_tokens)
            if not pairs:  # 如果没有字符对可以合并，提前结束
                print(f"第 {i} 轮：没有更多字符对可合并，提前结束")
                break
                
            # 找出频率最高的字符对
            best_pair = max(pairs, key=pairs.get)
            # 如果最高频率小于阈值，停止合并
            if pairs[best_pair] < self.min_frequency:
                print(f"第 {i} 轮：最高频率 {pairs[best_pair]} 低于阈值 {self.min_frequency}，停止合并")
                break
                
            print(f"第 {i+1} 轮合并：{best_pair} (频率: {pairs[best_pair]})")
            
            # 执行合并操作
            word_tokens = self.merge_vocab(best_pair, word_tokens)
            # 记录这次合并规则
            self.merges.append(best_pair)
        
        # 第六步：构建最终词汇表
        vocab = set()
        for word in word_tokens.keys():
            for symbol in word.split():
                vocab.add(symbol)
        
        print(f"最终词汇表大小：{len(vocab)}")
        
        # 将词汇表转换为{token: id}的格式
        self.vocab = {token: idx for idx, token in enumerate(sorted(vocab))}
        return self.vocab, self.merges


class BPETokenizer:
    """
    BPE分词器
    
    使用已训练的BPE模型进行文本编码和解码
    """
    
    def __init__(self, vocab: Dict[str, int] = None, merges: List[Tuple[str, str]] = None):
        
        self.vocab = vocab or {}  # 词汇表
        self.merges = merges or []  # 合并规则
        # 创建反向词汇表，用于解码：{id: token}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def bpe(self, word: str) -> str:
      
        # 将单词转换为字符序列，并添加结束标记
        word = ' '.join(word) + ' </w>'
        
        if len(word) == 1:  # 如果只有一个字符，直接返回
            return word
            
        # 获取当前单词中的所有字符对
        pairs = self.get_pairs(word)
        
        if not pairs:  # 如果没有字符对，直接返回
            return word
            
        # 开始迭代应用BPE合并规则
        while True:
            # 在当前字符对中，找出在合并规则中最早出现的那一对
            # 如果某个字符对不在合并规则中，返回无穷大的索引
            bigram = min(pairs, key=lambda pair: self.merges.index(pair) if pair in self.merges else float('inf'))
            
            # 如果这个字符对不在我们学习的合并规则中，停止合并
            if bigram not in self.merges:
                break
                
            # 执行合并操作
            first, second = bigram  # 获取要合并的两个字符
            new_word = []  # 新的字符序列
            i = 0
            
            # 遍历当前字符序列，查找并合并指定的字符对
            while i < len(word):
                try:
                    # 从当前位置开始查找第一个字符
                    j = word.index(first, i)
                    # 添加之前的所有字符
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 如果找不到第一个字符，添加剩余所有字符
                    new_word.extend(word[i:])
                    break
                    
                # 检查是否可以进行合并（当前是first，下一个是second）
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 合并这两个字符
                    new_word.append(first + second)
                    i += 2  # 跳过两个字符
                else:
                    # 如果不能合并，只添加当前字符
                    new_word.append(word[i])
                    i += 1
                    
            word = new_word  # 更新字符序列
            if len(word) == 1:  # 如果只剩一个token，结束合并
                break
            else:
                pairs = self.get_pairs(word)  # 重新获取字符对
                
        return ' '.join(word)  # 返回空格分隔的token序列
    
    def get_pairs(self, word):
       
        if isinstance(word, str):  # 如果是字符串，先分割成列表
            word = word.split()
        pairs = set()  # 使用集合避免重复
        prev_char = word[0]  # 前一个字符
        for char in word[1:]:  # 遍历后续字符
            pairs.add((prev_char, char))  # 添加字符对
            prev_char = char
        return pairs
    
    def encode(self, text: str) -> List[int]:
       
        tokens = []
        # 使用正则表达式分割文本为单词和标点
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        for word in words:
            # 对每个单词应用BPE编码
            bpe_tokens = self.bpe(word).split()
            for token in bpe_tokens:
                # 将token转换为ID
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    # 如果token不在词汇表中，使用未知词标记
                    tokens.append(self.vocab.get('<unk>', 0))
                    
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
       
        words = []
        for token in tokens:
            # 将ID转换为token
            if token in self.inverse_vocab:
                words.append(self.inverse_vocab[token])
            else:
                words.append('<unk>')  # 未知token
                
        # 拼接所有token
        text = ' '.join(words)
        # 处理结束标记：</w>表示单词结束，替换为空格
        text = text.replace('</w>', ' ')
        # 合并多个连续空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()  # 去除首尾空格


class BPESerializer:
    """
    BPE模型序列化工具
    
    提供模型和数据的保存/加载功能
    """
    
    @staticmethod
    def save_model(vocab: Dict[str, int], merges: List[Tuple[str, str]], filepath: str):
       
        model_data = {
            'vocab': vocab,    # 词汇表
            'merges': merges   # 合并规则
        }
        
        if filepath.endswith('.json'):
            # 保存为JSON格式（人类可读）
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)
        else:
            # 保存为二进制格式（更紧凑）
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
        
        if filepath.endswith('.json'):
            # 从JSON文件加载
            with open(filepath, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        else:
            # 从二进制文件加载
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
        return model_data['vocab'], [tuple(merge) for merge in model_data['merges']]
    
    @staticmethod
    def save_encoded_text(tokens: List[int], filepath: str):
        """保存编码后的文本"""
        with open(filepath, 'wb') as f:
            pickle.dump(tokens, f)
    
    @staticmethod 
    def load_encoded_text(filepath: str) -> List[int]:
        """加载编码后的文本"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_bpe_tokenizer(texts: List[str], vocab_size: int = 1000, min_frequency: int = 2) -> BPETokenizer:
  
    print("=== 开始训练BPE分词器 ===")
    
    # 创建词汇表构建器
    builder = BPEVocabBuilder(vocab_size, min_frequency)
    # 构建词汇表和合并规则
    vocab, merges = builder.build_vocab(texts)
    
    # 添加特殊标记
    vocab['<unk>'] = len(vocab)  # 未知词标记
    
    print("=== BPE分词器训练完成 ===")
    
    # 返回分词器实例
    return BPETokenizer(vocab, merges)


# 主程序：演示BPE分词器的使用
if __name__ == "__main__":
    print("BPE分词器演示程序")
    print("=" * 50)
    
    # 准备中文训练样本
    sample_texts = [
        "你好世界！这是一个测试文本。",
        "自然语言处理非常有用。",
        "我们正在从头构建词汇表。",
        "机器学习需要大量的训练数据。",
        "分词是文本预处理的第一步。"
    ]
    
    print("训练样本：")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    # 训练分词器
    print(f"\n开始训练BPE分词器（目标词汇表大小：100）")
    tokenizer = train_bpe_tokenizer(sample_texts, vocab_size=100)
    
    # 测试编码和解码
    test_text = "你好！这是一个测试。"
    print(f"\n测试文本：{test_text}")
    
    # 编码
    encoded = tokenizer.encode(test_text)
    print(f"编码结果：{encoded}")
    
    # 解码
    decoded = tokenizer.decode(encoded)
    print(f"解码结果：{decoded}")
    
    print(f"最终词汇表大小：{len(tokenizer.vocab)}")
    
    