import re
from collections import defaultdict, Counter


class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.pattern = re.compile(r"'\w+|\w+|[^\w\s]+")

    def preprocess_text(self, text):
        """预处理文本：转换为小写并添加单词结束符</w>"""
        words = self.pattern.findall(text.lower())
        return [word + '</w>' for word in words]

    def get_stats(self, words):
        """统计相邻符号对的频率"""
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def merge_vocab(self, words, pair):
        """合并指定的符号对"""
        new_words = []
        pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
        for word in words:
            new_word = pattern.sub(''.join(pair), word)
            new_words.append(new_word)
        return new_words

    def build_vocab(self, corpus):
        """构建BPE词表"""
        # 初始词表：所有字符加上</w>
        words = self.preprocess_text(corpus)
        vocab = Counter()
        for word in words:
            vocab.update(word.split())

        # 初始词表大小
        initial_vocab_size = len(vocab)
        print(f"初始词表大小: {initial_vocab_size}")

        # 如果目标词表大小小于初始词表大小，则无法继续合并
        if self.vocab_size <= initial_vocab_size:
            self.vocab = dict(vocab)
            return

        # 迭代合并符号对
        num_merges = self.vocab_size - initial_vocab_size
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break

            # 找到频率最高的符号对
            best_pair = max(pairs, key=pairs.get)

            # 合并符号对
            words = self.merge_vocab(words, best_pair)

            # 记录合并操作
            self.merges[best_pair] = i

            # 更新词表
            vocab = Counter()
            for word in words:
                vocab.update(word.split())

        # 构建最终词表
        self.vocab = dict(vocab)

    def encode_word(self, word):
        """编码单个单词"""
        word = word.lower() + '</w>'
        tokens = list(word)

        # 应用所有合并规则
        for pair, _ in sorted(self.merges.items(), key=lambda x: x[1], reverse=True):
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text):
        words = self.pattern.findall(text)
        encoded_tokens = []
        for word in words:
            encoded_tokens.extend(self.encode_word(word))
        return encoded_tokens

    def get_vocab(self):
        """获取词表"""
        return self.vocab

if __name__ == "__main__":
    import time

    bpe = BPE(vocab_size=60)

    corpus = "Hello world! Hello BPE tokenizer demo."

    print("原始语料：", corpus)
    print("\n=== 开始训练 BPE 词表 ===\n")
    words = bpe.preprocess_text(corpus)
    vocab = Counter()
    for word in words:
        vocab.update(word.split())
    print(f"初始词表大小: {len(vocab)}")
    print("初始词表：", dict(vocab))
    print("\n--- 合并过程 ---\n")

    # 构建词表并显示合并过程
    initial_vocab_size = len(vocab)
    num_merges = bpe.vocab_size - initial_vocab_size
    words_copy = words.copy()
    for i in range(num_merges):
        pairs = bpe.get_stats(words_copy)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        print(f"Step {i+1}: 合并最频繁的符号对 {best_pair}，频率={pairs[best_pair]}")
        words_copy = bpe.merge_vocab(words_copy, best_pair)
        # 显示前10个词的变化
        print("前10个词示例：", words_copy[:10])
        bpe.merges[best_pair] = i
        time.sleep(0.3)  # 让输出更容易跟上

    # 构建最终词表
    final_vocab = Counter()
    for word in words_copy:
        final_vocab.update(word.split())
    bpe.vocab = dict(final_vocab)

    print("\n=== 训练完成 ===")
    print("训练后的词表：", bpe.get_vocab())

    # 交互式编码
    print("\n=== 开始交互式编码 ===")
    while True:
        text = input("\n请输入要编码的文本（输入 'exit' 退出）：")
        if text.lower() == 'exit':
            break
        encoded = bpe.encode(text)
        print("编码结果：", encoded)
