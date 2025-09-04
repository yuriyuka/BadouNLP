import json
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        
    def get_stats(self, vocab):
        """统计词汇对的频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并词汇对"""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            w_out = p.sub(''.join(pair), word)
            new_vocab[w_out] = vocab[word]
        return new_vocab
    
    def preprocess_text(self, text):
        """预处理文本，添加空格分隔符"""
        # 在标点符号和中文字符之间添加空格
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        # 将中文字符逐个分开
        text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
        # 清理多余的空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fit(self, texts):
        """训练BPE模型"""
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # 构建初始词汇表
        vocab = defaultdict(int)
        for text in processed_texts:
            for word in text.split():
                # 为每个字符添加结束符
                word_with_end = ' '.join(list(word)) + ' </w>'
                vocab[word_with_end] += 1
        
        # 开始BPE训练
        # 正确计算需要合并的次数
        num_merges = self.vocab_size - len(set(char for word in vocab for char in word.split()))
        # 确保num_merges不会是负数
        num_merges = max(0, min(num_merges, self.vocab_size - 5))  # 减去特殊标记的数量
        
        print(f"开始BPE训练，将进行 {num_merges} 次合并")
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
                
            # 找到最频繁的词汇对
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            
            # 记录合并操作
            self.merges[i] = best_pair
            
            if num_merges > 0 and i % max(1, num_merges // 10) == 0:  # 每10%进度打印一次
                print(f"BPE训练进度: {i}/{num_merges}")
        
        # 构建最终的词汇表
        self.vocab = {}
        for word in vocab:
            for token in word.split():
                self.vocab[token] = self.vocab.get(token, 0) + 1
        
        # 添加特殊标记
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
        for token in special_tokens:
            self.vocab[token] = 1
        
        # 创建token到ID的映射
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        self.token_to_id = {token: idx for idx, (token, _) in enumerate(sorted_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"BPE训练完成，词汇表大小: {len(self.vocab)}")
    
    def encode(self, text):
        """将文本编码为token ID序列"""
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 初始化为字符级tokens
        tokens = []
        for word in processed_text.split():
            # 将单词分解为字符并添加结束符
            word_tokens = list(word) + ['</w>']
            
            # 应用BPE合并规则
            changed = True
            while changed:
                changed = False
                # 找到最佳合并对
                best_pair_idx = -1
                best_pair_rank = float('inf')
                
                # 查找可以合并的最佳对（根据合并顺序）
                for i in range(len(word_tokens)-1):
                    pair = (word_tokens[i], word_tokens[i+1])
                    # 查找这个对在合并历史中的位置
                    for rank, merge_pair in self.merges.items():
                        if pair == merge_pair and rank < best_pair_rank:
                            best_pair_rank = rank
                            best_pair_idx = i
                            break
                
                # 如果找到可以合并的对
                if best_pair_idx != -1:
                    pair = (word_tokens[best_pair_idx], word_tokens[best_pair_idx+1])
                    # 执行合并
                    word_tokens = (word_tokens[:best_pair_idx] + 
                                 [''.join(pair)] + 
                                 word_tokens[best_pair_idx+2:])
                    changed = True
            
            # 移除</w>标记并添加到结果中
            if word_tokens and word_tokens[-1] == '</w>':
                tokens.extend(word_tokens[:-1])
            else:
                tokens.extend(word_tokens)
        
        # 转换为ID
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id['<UNK>'])
        return token_ids
    
    def decode(self, token_ids):
        """将token ID序列解码为文本"""
        tokens = []
        for idx in token_ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                tokens.append(token)
            else:
                tokens.append('<UNK>')
        
        text = ''.join(tokens)
        # 移除空格（中文文本通常不需要空格分隔）
        text = text.replace(' ', '')
        return text

def load_json_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    # 从JSON数据中提取文本
    for key in data:
        for item in data[key]:
            for field in item:
                if item[field]:
                    # 去除引号
                    text = item[field].strip('"')
                    texts.append(text)
    
    return texts

# 主程序
if __name__ == "__main__":
    # 加载测试数据
    file_path = r"F:\BaiduNetdiskDownload\八斗精品班\第十四周\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\data\测试数据.json"
    texts = load_json_data(file_path)
    
    print(f"加载了 {len(texts)} 条文本数据")
    
    # 创建BPE分词器
    tokenizer = BPETokenizer(vocab_size=3000)
    
    # 训练BPE模型，使用所有数据
    print("开始训练BPE模型...")
    tokenizer.fit(texts)  # 使用所有数据进行训练
    
    # 测试编码和解码
    test_text = "消防稳压泵无法供水"
    print(f"\n测试文本: {test_text}")
    
    # 编码
    token_ids = tokenizer.encode(test_text)
    print(f"编码结果: {token_ids}")
    
    # 解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码结果: {decoded_text}")
    
    # 显示词汇表示例
    print(f"\n词汇表示例 (前20个):")
    for i, (token, id_) in enumerate(list(tokenizer.token_to_id.items())[:20]):
        print(f"  {token}: {id_}")