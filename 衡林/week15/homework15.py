import os
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    if len(ids) < 2:
        return {}
    
    counts = Counter()
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] += 1
    return counts


def merge_pair(pair: Tuple[int, int], ids: List[int], idx: int) -> List[int]:
    new_ids = []
    i = 0
    n = len(ids)
    
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self, vocab_size: int = 256):
        if vocab_size < 256:
            raise ValueError("Vocabulary size must be at least 256")
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}  
        self.merge_rules: Dict[Tuple[int, int], int] = {}
        self.reverse_vocab: Dict[int, Tuple[int, int]] = {} 
    def train(self, text: str, verbose: bool = True) -> None:
        """
        训练BPE分词器
        
        Args:
            text: 训练文本
            verbose: 是否显示训练进度
        """
        if not text:
            raise ValueError("Training text cannot be empty")
            
        num_merges = self.vocab_size - 256
        ids = list(text.encode("utf-8"))
        
        # 初始化基础词汇表
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merge_rules = {}
        
        start_time = time.time()
        
        for i in range(num_merges):
            # 统计频率
            counts = get_stats(ids)
            if not counts:
                if verbose:
                    print(f"Early stopping: no more pairs to merge after {i} merges")
                break
                
            # 找到最高频的符号对
            pair, count = counts.most_common(1)[0]
            new_id = 256 + i
            
            # 执行合并
            ids = merge_pair(pair, ids, new_id)
            self.merge_rules[pair] = new_id
            self.reverse_vocab[new_id] = pair
            
            if verbose and i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Merge {i+1}/{num_merges}: "
                      f"pair={pair} -> {new_id} (count={count}), "
                      f"time={elapsed:.2f}s")
        
        if verbose:
            total_time = time.time() - start_time
            print(f"Training completed in {total_time:.2f} seconds")
            print(f"Final vocabulary size: {256 + len(self.merge_rules)}")

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为BPE token序列
        
        Args:
            text: 输入文本
            
        Returns:
            token序列
        """
        if not text:
            return []
            
        # 初始化为字节序列
        ids = list(text.encode("utf-8"))
        
        # 应用所有合并规则，直到无法继续合并
        changed = True
        while changed and len(ids) >= 2:
            changed = False
            counts = get_stats(ids)
            
            # 按合并规则的应用顺序尝试合并（频率高的优先）
            for pair in sorted(counts.keys(), 
                             key=lambda p: self.merge_rules.get(p, float('inf'))):
                if pair in self.merge_rules:
                    new_id = self.merge_rules[pair]
                    new_ids = merge_pair(pair, ids, new_id)
                    if len(new_ids) < len(ids):  # 如果发生了合并
                        ids = new_ids
                        changed = True
                        break  # 重新开始统计
        
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        将BPE token序列解码为文本
        
        Args:
            ids: token序列
            
        Returns:
            解码后的文本
        """
        if not ids:
            return ""
            
        # 逐步展开合并的token
        current_ids = ids[:]
        
        while True:
            new_ids = []
            changed = False
            
            for token_id in current_ids:
                if token_id in self.reverse_vocab:
                    # 展开合并的token
                    left, right = self.reverse_vocab[token_id]
                    new_ids.extend([left, right])
                    changed = True
                else:
                    new_ids.append(token_id)
            
            if not changed:
                break
            current_ids = new_ids
        
        # 转换为字节并解码
        try:
            byte_data = bytes(current_ids)
            return byte_data.decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Decoding error: {e}")

    def save_vocab(self, filepath: str) -> None:
        """保存词汇表到文件"""
        import json
        # 转换为可序列化的格式
        vocab_data = {
            'vocab_size': self.vocab_size,
            'merge_rules': {f"{k[0]},{k[1]}": v for k, v in self.merge_rules.items()},
            'reverse_vocab': {k: list(v) for k, v in self.reverse_vocab.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filepath: str) -> None:
        """从文件加载词汇表"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab_size = vocab_data['vocab_size']
        self.merge_rules = {
            tuple(map(int, k.split(','))): v 
            for k, v in vocab_data['merge_rules'].items()
        }
        self.reverse_vocab = {
            int(k): tuple(v) 
            for k, v in vocab_data['reverse_vocab'].items()
        }


def process_large_corpus(directory: str, max_files: Optional[int] = None) -> str:
    """
    处理大语料库，支持分批读取
    
    Args:
        directory: 包含文本文件的目录
        max_files: 最大处理文件数（None表示处理所有文件）
        
    Returns:
        合并后的文本
    """
    all_text = []
    file_count = 0
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    all_text.append(text)
                    file_count += 1
                    
                    if max_files and file_count >= max_files:
                        break
                        
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
    
    return "".join(all_text)


if __name__ == "__main__":
    try:
        # 初始化分词器
        tokenizer = BPETokenizer(vocab_size=1000)  # 较小的词汇表用于演示
        
        # 处理语料库
        print("Processing corpus...")
        corpus_text = process_large_corpus("Heroes", max_files=10)  # 限制文件数用于演示
        
        if not corpus_text:
            print("No text found in corpus directory")
        else:
            # 训练分词器
            print("Training tokenizer...")
            tokenizer.train(corpus_text, verbose=True)
            
            # 测试编码解码
            test_text = "This is the Hugging Face Course..."
            print(f"\nOriginal text: {test_text}")
            
            encoded = tokenizer.encode(test_text)
            print(f"Encoded: {encoded}")
            print(f"Number of tokens: {len(encoded)}")
            print(f"Compression ratio: {len(test_text.encode('utf-8')) / len(encoded):.2f}")
            
            decoded = tokenizer.decode(encoded)
            print(f"Decoded: {decoded}")
            print(f"Match: {test_text == decoded}")
            
            # 保存词汇表
            tokenizer.save_vocab("bpe_vocab.json")
            print("Vocabulary saved to bpe_vocab.json")
            
    except Exception as e:
        print(f"Error: {e}")
