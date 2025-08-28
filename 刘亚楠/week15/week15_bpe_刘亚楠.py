import re
from collections import defaultdict, Counter

class BPE:
    def __init__(self, vocab_size=1000, end_of_word="</w>"):
        self.vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.vocab = {}  # token -> id
        self.merges = []  # 存储合并规则
        self.inverse_vocab = {}  # id -> token
    
    def preprocess(self, text):
        """预处理文本：分词并添加结束符号"""
        words = re.findall(r'\w+', text.lower())
        return [word + self.end_of_word for word in words]
    
    def get_stats(self, vocab):
        """统计字节对频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并字节对"""
        first, second = pair
        new_vocab = {}
        pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
        
        for word, freq in vocab.items():
            # 将匹配的字节对合并
            new_word = pattern.sub(first + second, word)
            new_vocab[new_word] = freq
        
        return new_vocab
    
    def train(self, corpus):
        """训练BPE模型，构建词表"""
        # 初始词汇统计（单词频率）
        words = self.preprocess(corpus)
        word_freq = Counter(words)
        
        # 初始词表：所有字符 + 结束符
        base_vocab = set()
        for word in word_freq.keys():
            for char in word:
                base_vocab.add(char)
        
        # 初始化当前词汇表（用空格分隔字符）
        current_vocab = {}
        for word, freq in word_freq.items():
            # 将单词拆分为字符并用空格连接
            tokenized_word = ' '.join(list(word))
            current_vocab[tokenized_word] = freq
        
        # 构建基础词表
        self.vocab = {token: idx for idx, token in enumerate(sorted(base_vocab))}
        
        # 开始迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # 统计字节对频率
            pairs = self.get_stats(current_vocab)
            if not pairs:
                break
                
            # 找到最频繁的字节对
            best_pair = max(pairs, key=pairs.get)
            
            # 记录合并规则
            self.merges.append(best_pair)
            
            # 合并字节对
            current_vocab = self.merge_vocab(best_pair, current_vocab)
            
            # 将新token加入词表
            new_token = best_pair[0] + best_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
            
            print(f"Merge {i+1}: {best_pair} -> {new_token}")
        
        # 构建反向词表
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        return self.vocab, self.merges
    
    def tokenize_word(self, word):
        """将单个单词tokenize"""
        # 添加结束符并拆分为字符
        word = word + self.end_of_word
        tokens = list(word)
        
        # 应用所有合并规则
        for merge in self.merges:
            first, second = merge
            new_token = first + second
            
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == first and tokens[i + 1] == second:
                    # 合并字节对
                    tokens[i] = new_token
                    del tokens[i + 1]
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text):
        """将文本编码为token ID序列"""
        words = re.findall(r'\w+', text.lower())
        token_ids = []
        
        for word in words:
            tokens = self.tokenize_word(word)
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # 处理未知token（这里简单跳过，实际应用中可能需要特殊处理）
                    print(f"Warning: Unknown token '{token}'")
        
        return token_ids
    
    def decode(self, token_ids):
        """将token ID序列解码为文本"""
        tokens = [self.inverse_vocab.get(token_id, '<?>') for token_id in token_ids]
        
        # 合并tokens并移除结束符
        text = ''.join(tokens)
        text = text.replace(self.end_of_word, ' ')
        
        return text.strip()

# 测试代码
if __name__ == "__main__":
    # 示例语料
    corpus = "low low low low low low lowest lowest newer newer newer newer wider wider"
    
    # 初始化BPE
    bpe = BPE(vocab_size=50)
    
    # 训练模型
    print("Training BPE model...")
    vocab, merges = bpe.train(corpus)
    print(f"\nFinal vocabulary size: {len(vocab)}")
    print("Top 20 tokens:", list(vocab.keys())[:20])
    print("Merges:", merges[:10])
    
    # 测试编码
    test_text = "lower wider newest"
    print(f"\nTesting encoding for: '{test_text}'")
    
    token_ids = bpe.encode(test_text)
    print("Token IDs:", token_ids)
    
    # 测试解码
    decoded_text = bpe.decode(token_ids)
    print("Decoded text:", decoded_text)
    
    # 显示详细的tokenization过程
    print(f"\nDetailed tokenization of '{test_text}':")
    words = re.findall(r'\w+', test_text.lower())
    for word in words:
        tokens = bpe.tokenize_word(word)
        print(f"'{word}' -> {tokens}")
