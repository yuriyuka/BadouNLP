import re
import os
from collections import defaultdict, Counter
import glob

class FileBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.special_tokens = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}
        
    def read_files_from_directory(self, directory_path, file_extensions=None):
        """
        从目录读取所有指定格式的文件内容
        
        Args:
            directory_path: 目录路径
            file_extensions: 文件扩展名列表，如 ['txt', 'csv', 'py']
        
        Returns:
            所有文件内容的拼接字符串
        """
        if file_extensions is None:
            file_extensions = ['txt']
        
        all_text = []
        
        # 构建文件搜索模式
        patterns = []
        for ext in file_extensions:
            if ext.startswith('.'):
                patterns.append(os.path.join(directory_path, f"*{ext}"))
            else:
                patterns.append(os.path.join(directory_path, f"*.{ext}"))
        
        # 查找所有匹配的文件
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(pattern))
        
        print(f"找到 {len(file_paths)} 个文件:")
        for file_path in file_paths:
            print(f"  - {os.path.basename(file_path)}")
        
        # 读取文件内容
        total_size = 0
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_text.append(content)
                    total_size += len(content)
                    print(f"已读取: {os.path.basename(file_path)} ({len(content)} 字符)")
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                        all_text.append(content)
                        total_size += len(content)
                        print(f"已读取(GBK): {os.path.basename(file_path)} ({len(content)} 字符)")
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
            except Exception as e:
                print(f"无法读取文件 {file_path}: {e}")
        
        print(f"总共读取 {total_size} 个字符")
        return '\n'.join(all_text)
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        # 保留基本的标点符号
        text = re.sub(r'([.!?,;:()\"\'])', r' \1 ', text)
        # 处理连续空格
        text = re.sub(r'\s+', ' ', text)
        # 添加单词边界标记
        text = re.sub(r'(\w+)', r'▁\1', text)
        return text.strip()
    
    def get_stats(self, vocab):
        """统计符号对频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        """合并符号对"""
        first, second = pair
        new_vocab = {}
        pattern = re.compile(r'(?<!\S)' + re.escape(f'{first} {second}') + r'(?!\S)')
        
        for word, freq in vocab.items():
            new_word = pattern.sub(f'{first}{second}', word)
            new_vocab[new_word] = freq
        return new_vocab
    
    def train_from_directory(self, directory_path, file_extensions=None):
        """
        从目录中的文件训练BPE模型
        
        Args:
            directory_path: 包含训练文件的目录路径
            file_extensions: 要读取的文件扩展名列表
        """
        # 读取所有文件内容
        print("正在读取文件...")
        corpus = self.read_files_from_directory(directory_path, file_extensions)
        
        if not corpus.strip():
            raise ValueError("没有读取到任何文本内容，请检查目录路径和文件格式")
        
        # 预处理文本
        print("预处理文本...")
        preprocessed = self.preprocess_text(corpus)
        
        # 初始化词汇表
        vocab = Counter()
        words = preprocessed.split()
        print(f"处理 {len(words)} 个单词...")
        
        for word in words:
            if word.startswith('▁'):
                chars = list(word)
                tokenized = ' '.join(chars)
            else:
                tokenized = ' '.join(list(word))
            vocab[tokenized] += 1
        
        # 添加特殊标记到词汇表
        self.vocab.update(self.special_tokens)
        
        # 获取基础字符
        base_chars = set()
        for word in vocab:
            base_chars.update(word.split())
        
        # 初始化词汇表
        for i, char in enumerate(sorted(base_chars), start=len(self.special_tokens)):
            self.vocab[char] = i
        
        # BPE合并
        num_merges = self.vocab_size - len(self.vocab)
        self.merges = []
        
        print(f"开始BPE训练，目标词表大小: {self.vocab_size}")
        print(f"基础字符数: {len(base_chars)}")
        print(f"需要合并次数: {num_merges}")
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                print("没有更多可合并的符号对，提前终止")
                break
                
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]
            
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            # 添加新合并的符号到词汇表
            merged = ''.join(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            
            if (i + 1) % 10 == 0 or i == 0 or i == num_merges - 1:
                print(f"Merge {i+1}: {best_pair} -> '{merged}' (频率: {best_freq})")
        
        print(f"\n训练完成！最终词表大小: {len(self.vocab)}")
        return self.vocab
    
    def tokenize(self, text):
        """文本标记化"""
        text = self.preprocess_text(text)
        words = text.split()
        tokens = []
        
        for word in words:
            if word.startswith('▁'):
                current = list(word)
            else:
                current = list(word)
            
            # 应用所有合并规则
            for pair in self.merges:
                first, second = pair
                new_current = []
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and current[i] == first and current[i + 1] == second:
                        new_current.append(first + second)
                        i += 2
                    else:
                        new_current.append(current[i])
                        i += 1
                current = new_current
            
            tokens.extend(current)
        
        return tokens
    
    def encode(self, text):
        """编码文本到ID"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens]
    
    def decode(self, ids):
        """将ID解码回文本"""
        # 创建反向词汇表映射
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for id_val in ids:
            token = id_to_token.get(id_val, '<unk>')
            tokens.append(token)
        
        # 重建文本
        text = ''.join(tokens)
        text = text.replace('▁', ' ')  # 将边界标记转换为空格
        text = re.sub(r'\s+', ' ', text)  # 规范化空格
        return text.strip()
    
    def save_vocab(self, file_path):
        """保存词汇表到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
        print(f"词汇表已保存到: {file_path}")
    
    def load_vocab(self, file_path):
        """从文件加载词汇表"""
        self.vocab = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                self.vocab[token] = int(idx)
        print(f"已加载词汇表，大小: {len(self.vocab)}")

def main():
    # 创建BPE实例
    bpe = FileBPE(vocab_size=500)
    
    # 指定包含训练文件的目录
    data_directory = "./data"  # 修改为你的数据目录路径
    
    # 训练BPE模型（读取所有txt文件）
    try:
        vocab = bpe.train_from_directory(
            directory_path=data_directory,
            file_extensions=['txt', 'csv', 'py']  # 可以指定多种文件格式
        )
        
        # 保存词汇表
        bpe.save_vocab("bpe_vocab.txt")
        
        # 测试
        test_texts = [
            "hello world this is a test",
            "自然语言处理很有趣",
            "BPE算法用于文本tokenization"
        ]
        
        print("\n测试结果:")
        for test_text in test_texts:
            print(f"\n原文: '{test_text}'")
            tokens = bpe.tokenize(test_text)
            print(f"标记: {tokens}")
            ids = bpe.encode(test_text)
            print(f"ID序列: {ids}")
            decoded = bpe.decode(ids)
            print(f"解码: '{decoded}'")
            
    except Exception as e:
        print(f"错误: {e}")
        print("请确保目录路径正确且包含可读的文本文件")

if __name__ == "__main__":
    main()
