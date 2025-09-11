import os
from collections import defaultdict, Counter

class SimpleBPE:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.inverse_vocab = {}
    
    def read_files(self, folder_path):
        """读取文件夹中的所有txt文件内容"""
        text = ""
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    text += f.read().lower() + " "
        return text
    
    def get_stats(self, vocab):
        """获取符号对频率统计"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并符号对"""
        first, second = pair
        new_vocab = {}
        bigram = first + ' ' + second
        merged = first + second
        
        for word in vocab:
            new_word = word.replace(bigram, merged)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def train(self, folder_path):
        """训练BPE模型"""
        text = self.read_files(folder_path)
        words = text.split()
        
        # 初始词汇表
        vocab = Counter()
        for word in words:
            tokenized = ' '.join(list(word)) + ' </w>'
            vocab[tokenized] += 1
        
        # 基础词汇（所有字符）
        base_vocab = set()
        for word in vocab:
            base_vocab.update(word.split())
        
        # 开始合并
        num_merges = self.vocab_size - len(base_vocab)
        for _ in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            vocab = self.merge_vocab(best_pair, vocab)
        
        # 构建最终词汇表
        final_vocab = set()
        for word in vocab:
            final_vocab.update(word.split())
        
        # 创建词汇映射
        self.vocab = {token: idx for idx, token in enumerate(final_vocab)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        """编码文本"""
        words = text.lower().split()
        tokens = []
        token_ids = []
        
        for word in words:
            # 初始化为字符
            current_tokens = list(word) + ['</w>']
            
            # 应用合并规则
            for merge in self.merges:
                new_tokens = []
                i = 0
                while i < len(current_tokens):
                    if (i < len(current_tokens) - 1 and 
                        current_tokens[i] == merge[0] and 
                        current_tokens[i+1] == merge[1]):
                        new_tokens.append(merge[0] + merge[1])
                        i += 2
                    else:
                        new_tokens.append(current_tokens[i])
                        i += 1
                current_tokens = new_tokens
            
            tokens.extend(current_tokens)
            for token in current_tokens:
                token_ids.append(self.vocab.get(token, -1))
        
        return tokens, token_ids
    
    def decode(self, token_ids):
        """解码token IDs"""
        tokens = [self.inverse_vocab.get(idx, '<?>') for idx in token_ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

def main():
    # 创建示例文本文件
    os.makedirs("texts", exist_ok=True)
    with open("texts/sample1.txt", "w") as f:
        f.write("hello world this is a test")
    with open("texts/sample2.txt", "w") as f:
        f.write("the quick brown fox jumps")
    
    # 训练BPE
    bpe = SimpleBPE(vocab_size=100)
    bpe.train("texts")
    
    print("词汇表大小:", len(bpe.vocab))
    print("前10个词汇:", list(bpe.vocab.items())[:10])
    
    # 编码测试
    test_text = "hello world test"
    tokens, token_ids = bpe.encode(test_text)
    print(f"\n编码 '{test_text}':")
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    
    # 解码测试
    decoded = bpe.decode(token_ids)
    print(f"解码结果: '{decoded}'")

if __name__ == "__main__":
    main()
