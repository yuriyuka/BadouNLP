import re
from collections import defaultdict, Counter


class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merge_rules = []

    def get_stats(self, vocab):
        """统计字符对的出现频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """合并指定的字符对"""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        for word in vocab:
            w_out = p.sub(''.join(pair), word)
            new_vocab[w_out] = vocab[word]
        return new_vocab

    def train(self, text):
        """训练BPE模型"""
        # 初始化词汇表（将文本分割为字符并添加结束符</w>）
        vocab = Counter()
        words = text.strip().split()
        for word in words:
            # 将单词分割为字符并添加结束符号
            token = ' '.join(list(word)) + ' </w>'
            vocab[token] += 1

        # 初始词汇表（所有字符）
        self.initial_vocab = set()
        for word in vocab:
            for char in word.split():
                self.initial_vocab.add(char)

        # 逐步合并最常见的字符对
        while len(self.initial_vocab) + len(self.merge_rules) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            # 选择最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            self.merge_rules.append(best_pair)
            vocab = self.merge_vocab(best_pair, vocab)

        # 构建最终词汇表
        self.vocab = self.initial_vocab.copy()
        for rule in self.merge_rules:
            self.vocab.add(''.join(rule))

    def encode_word(self, word):
        """编码单个单词"""
        # 初始分割为字符
        tokens = list(word) + ['</w>']

        # 应用所有合并规则
        for pair in self.merge_rules:
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
        """编码文本"""
        words = text.strip().split()
        encoded_tokens = []
        for word in words:
            encoded_tokens.extend(self.encode_word(word))
        return encoded_tokens

    def get_vocab(self):
        """返回词汇表"""
        return self.vocab


# 示例使用
if __name__ == "__main__":
    # 示例文本
    text = "low lower newest widest"

    # 初始化BPE模型，设置目标词汇表大小为30
    bpe = BPE(vocab_size=30)

    # 训练模型
    bpe.train(text)

    # 获取词汇表
    vocab = bpe.get_vocab()
    print("词汇表:", sorted(vocab))
    print("词汇表大小:", len(vocab))

    # 编码文本
    encoded_text = bpe.encode("low lower newest widest")
    print("编码后的文本:", encoded_text)

    # 编码新文本
    new_text = "lowest"
    encoded_new = bpe.encode(new_text)
    print(f"'{new_text}' 编码为:", encoded_new)
