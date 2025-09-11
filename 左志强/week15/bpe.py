import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class BPE:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.inverse_merges = {}
        self.special_tokens = {}
    
    def add_special_tokens(self, tokens: List[str]):
        """添加特殊标记"""
        for token in tokens:
            self.special_tokens[token] = len(self.vocab) + len(self.special_tokens)
    
    def get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """统计字符对频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """合并字符对"""
        first, second = pair
        new_vocab = {}
        pattern = re.compile(r'(?<!\S)' + re.escape(first + second) + r'(?!\S)')
        
        for word in vocab:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = vocab[word]
        return new_vocab
    
    def train(self, text: str, verbose: bool = False):
        """训练BPE模型"""
        # 初始化词汇表（字符级别）
        words = text.split()
        vocab = defaultdict(int)
        for word in words:
            # 将单词转换为字符元组，并在末尾添加</w>
            chars = tuple(list(word) + ['</w>'])
            vocab[chars] += 1
        
        # 初始词汇表（所有字符）
        base_vocab = set()
        for word in vocab:
            for char in word:
                base_vocab.add(char)
        
        # 添加特殊标记到词汇表
        for token in self.special_tokens:
            base_vocab.add(token)
        
        # 合并操作
        merges = {}
        num_merges = self.vocab_size - len(base_vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]
            
            if best_freq < 2:  # 如果最高频率小于2，停止合并
                break
                
            if verbose:
                print(f"Step {i+1}: Merging {best_pair} with frequency {best_freq}")
            
            # 更新词汇表
            vocab = self.merge_vocab(best_pair, vocab)
            
            # 记录合并操作
            merges[best_pair] = best_pair[0] + best_pair[1]
            self.merges[best_pair] = i + len(base_vocab)
            self.inverse_merges[best_pair[0] + best_pair[1]] = best_pair
        
        # 构建最终词汇表
        self.vocab = {token: idx for idx, token in enumerate(base_vocab)}
        
        # 添加合并后的token到词汇表
        for merge in merges.values():
            if merge not in self.vocab:
                self.vocab[merge] = len(self.vocab)
        
        # 添加特殊标记
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
    
    def tokenize_word(self, word: str) -> List[str]:
        """将单词分词为BPE tokens"""
        # 处理特殊标记
        for special_token in self.special_tokens:
            if special_token in word:
                # 这里简化处理，实际应用中可能需要更复杂的逻辑
                return [special_token] if word == special_token else [word]
        
        # 初始化为字符列表
        tokens = list(word) + ['</w>']
        
        # 应用合并规则
        while len(tokens) > 1:
            # 查找最频繁的字符对
            pairs = set()
            for i in range(len(tokens) - 1):
                pairs.add((tokens[i], tokens[i+1]))
            
            # 查找可合并的字符对
            merge_candidate = None
            for pair in pairs:
                if pair in self.merges:
                    if merge_candidate is None or self.merges[pair] < self.merges[merge_candidate]:
                        merge_candidate = pair
            
            if merge_candidate is None:
                break
                
            # 执行合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge_candidate:
                    new_tokens.append(merge_candidate[0] + merge_candidate[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs"""
        words = text.split()
        token_ids = []
        
        for word in words:
            tokens = self.tokenize_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # 处理未知token（这里简单地跳过）
                    pass
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token IDs解码为文本"""
        # 创建反向词汇表
        id_to_token = {id: token for token, id in self.vocab.items()}
        
        tokens = [id_to_token.get(id, '<unk>') for id in token_ids]
        text = ''.join(tokens)
        
        # 处理结束标记
        text = text.replace('</w>', ' ')
        
        return text.strip()

# 示例用法
if __name__ == "__main__":
    # 示例文本
    text = "low lower lowest new newer newest wide wider widest"
    
    # 初始化BPE
    bpe = BPE(vocab_size=50)
    
    # 添加特殊标记
    bpe.add_special_tokens(["<unk>", "<pad>", "<s>", "</s>"])
    
    # 训练BPE模型
    bpe.train(text, verbose=True)
    
    # 打印词汇表
    print("\nVocabulary:")
    for token, idx in sorted(bpe.vocab.items(), key=lambda x: x[1]):
        print(f"{idx}: {repr(token)}")
    
    # 编码文本
    test_text = "lower widest newer"
    encoded = bpe.encode(test_text)
    print(f"\nEncoded '{test_text}': {encoded}")
    
    # 解码
    decoded = bpe.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    # 显示分词结果
    print("\nTokenization:")
    for word in test_text.split():
        tokens = bpe.tokenize_word(word)
        print(f"{word} -> {tokens}")
