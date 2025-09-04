import re
import os
from collections import defaultdict, Counter
import glob


class FileBPE:
    def __init__(self, vocab_size=1000):
        # 初始化BPE模型参数
        self.vocab_size = vocab_size  # 目标词汇表大小
        self.vocab = {}  # 词汇表，存储token到ID的映射
        self.merges = []  # 存储合并历史的列表
        # 特殊标记及其ID
        self.special_tokens = {'<unk>': 0, '<pad>': 1, '<s>': 2, '</s>': 3}

    def read_files_from_directory(self, directory_path, file_extensions=None):
        # 若未指定文件扩展名，默认读取txt文件
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
        # 转换为小写
        text = text.lower()
        # 为标点符号添加空格，方便后续处理
        text = re.sub(r'([.!?,;:()\"\'])', r' \1 ', text)
        # 处理连续空格，合并为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 为单词添加边界标记▁
        text = re.sub(r'(\w+)', r'▁\1', text)
        return text.strip()

    def get_stats(self, vocab):
        # 统计所有符号对的出现频率
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        # 合并词汇表中的符号对
        first, second = pair
        new_vocab = {}
        # 编译正则表达式，用于匹配要合并的符号对
        pattern = re.compile(r'(?<!\S)' + re.escape(f'{first} {second}') + r'(?!\S)')

        for word, freq in vocab.items():
            # 替换符号对为合并后的符号
            new_word = pattern.sub(f'{first}{second}', word)
            new_vocab[new_word] = freq
        return new_vocab

    def train_from_directory(self, directory_path, file_extensions=None):
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

        # 获取基础字符集合
        base_chars = set()
        for word in vocab:
            base_chars.update(word.split())

        # 初始化词汇表ID
        for i, char in enumerate(sorted(base_chars), start=len(self.special_tokens)):
            self.vocab[char] = i

        # 计算需要合并的次数
        num_merges = self.vocab_size - len(self.vocab)
        self.merges = []

        print(f"开始BPE训练，目标词表大小: {self.vocab_size}")
        print(f"基础字符数: {len(base_chars)}")
        print(f"需要合并次数: {num_merges}")

        # 执行BPE合并过程
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                print("没有更多可合并的符号对，提前终止")
                break

            # 选择出现频率最高的符号对
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            # 合并符号对
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            # 添加新合并的符号到词汇表
            merged = ''.join(best_pair)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            # 定期输出合并进度
            if (i + 1) % 10 == 0 or i == 0 or i == num_merges - 1:
                print(f"Merge {i + 1}: {best_pair} -> '{merged}' (频率: {best_freq})")

        print(f"\n训练完成！最终词表大小: {len(self.vocab)}")
        return self.vocab

    def tokenize(self, text):
        # 对文本进行预处理
        text = self.preprocess_text(text)
        words = text.split()
        tokens = []

        # 对每个词应用BPE合并规则
        for word in words:
            if word.startswith('▁'):
                current = list(word)
            else:
                current = list(word)

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
        # 将文本转换为token序列
        tokens = self.tokenize(text)
        # 将token转换为ID，未知token用<unk>的ID
        return [self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens]

    def decode(self, ids):
        # 创建ID到token的映射
        id_to_token = {v: k for k, v in self.vocab.items()}

        tokens = []
        for id_val in ids:
            token = id_to_token.get(id_val, '<unk>')
            tokens.append(token)

        # 将token序列转换回文本
        text = ''.join(tokens)
        text = text.replace('▁', ' ')  # 将边界标记转换为空格
        text = re.sub(r'\s+', ' ', text)  # 规范化空格
        return text.strip()

    def save_vocab(self, file_path):
        # 保存词汇表到文件
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
        print(f"词汇表已保存到: {file_path}")

    def load_vocab(self, file_path):
        # 从文件加载词汇表
        self.vocab = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                self.vocab[token] = int(idx)
        print(f"已加载词汇表，大小: {len(self.vocab)}")


def main():
    # 创建BPE实例，设置词汇表大小为500
    bpe = FileBPE(vocab_size=500)
    # 指定包含训练文件的目录
    data_directory = "./data"
    # 训练BPE模型（读取txt、csv、py文件）
    try:
        vocab = bpe.train_from_directory(
            directory_path=data_directory,
            file_extensions=['txt', 'csv', 'py']
        )

        # 保存词汇表
        bpe.save_vocab("bpe_vocab.txt")

        # 测试BPE的分词、编码和解码功能
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
