#week15作业
#尝试用bpe完成词表构建和文本的序列化。

import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class BPE:
    def __init__(self, vocab_size: int = 1000, end_of_word: str = "</w>"):
        self.vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.vocab = defaultdict(int)
        self.merges = {}
        self.reverse_merges = {}
    
    def get_stats(self, words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """统计字符对的出现频率"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """合并字符对并更新词汇表"""
        new_words = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        return new_words
    
    def train(self, text: str):
        """训练BPE模型，构建词表"""
        # 初始词表：字符 + 频率统计
        words = Counter(text.split())
        words = {f"{' '.join(word)}{self.end_of_word}": freq for word, freq in words.items()}
        
        # 初始化基础词表（单个字符）
        self.vocab = Counter()
        for word, freq in words.items():
            symbols = word.split()
            self.vocab.update(symbols)
        
        # 逐步合并直到达到目标词表大小
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break
            
            # 选择频率最高的字符对
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:  # 不再有可合并的字符对
                break
            
            # 执行合并
            words = self.merge_vocab(best_pair, words)
            
            # 记录合并规则
            merged_token = "".join(best_pair)
            self.merges[best_pair] = merged_token
            self.reverse_merges[merged_token] = best_pair
            
            # 更新词表
            self.vocab[merged_token] = pairs[best_pair]
    
    def encode_word(self, word: str) -> List[str]:
        """将单个单词编码为子词序列"""
        word = f"{word}{self.end_of_word}"
        tokens = [char for char in word[:-1]] + [word[-1] + self.end_of_word]
        
        # 应用所有合并规则
        while True:
            pairs = self.get_stats({tuple_to_str(tokens): 1})
            if not pairs:
                break
            
            # 查找当前可用的最长合并规则
            best_pair = None
            for pair in self.merges:
                if pair in pairs:
                    best_pair = pair
                    break
            
            if not best_pair:
                break
            
            # 执行合并
            merged = self.merges[best_pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens

    def encode(self, text: str) -> List[str]:
        """将文本编码为子词序列"""
        words = text.split()
        encoded_tokens = []
        for word in words:
            tokens = self.encode_word(word)
            encoded_tokens.extend(tokens)
        return encoded_tokens
    
    def get_vocabulary(self) -> Set[str]:
        """获取完整词表"""
        full_vocab = set()
        for token in self.vocab:
            full_vocab.add(token)
            # 添加所有基础字符
            if token.endswith(self.end_of_word):
                full_vocab.add(token.replace(self.end_of_word, ""))
        return full_vocab

def tuple_to_str(tokens: List[str]) -> str:
    """将token列表转换为字符串表示"""
    return " ".join(tokens)

# 用法
if __name__ == "__main__":
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    
    # 假设文本数据
    text = "low low low low lower lowest newer newer newer newest widest"
    
    # 初始化并训练BPE
    bpe = BPE(vocab_size=20)
    bpe.train(text)
    
    # 获取词表
    vocabulary = bpe.get_vocabulary()
    print(f"BPE词表 ({len(vocabulary)}个):")
    print(sorted(vocabulary))
    
    # 测试编码
    test_word = "lowest"
    encoded = bpe.encode_word(test_word)
    print(f"\n单词 '{test_word}' 编码为: {encoded}")
    
    # 完整文本编码
    full_text = "lowest newest unknown"
    encoded_text = bpe.encode(full_text)
    print(f"\n文本 '{full_text}' 编码为: {encoded_text}")

    # 转换为PyTorch张量（示意）
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    input_ids = [token_to_id.get(token, -1) for token in encoded_text]
    tensor = torch.tensor(input_ids, dtype=torch.long)
    print("\nPyTorch张量表示:")
    print(tensor)
