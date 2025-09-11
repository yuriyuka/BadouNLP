import re

class BPE:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word_counts = {}
        self.merge_rules = {}

    # 找相邻字符对
    def get_pairs(self, chars):
        pairs = set()
        for i in range(len(chars) - 1):
            pairs.add((chars[i], chars[i + 1]))
        return pairs

    # 初始化词表
    def build_vocab(self, corpus):
        vocab = {}
        for word in corpus:
            chars = list(word) + ['</w>']
            for c in chars:
                if c in vocab:
                    vocab[c] += 1
                else:
                    vocab[c] = 1
            key = ' '.join(chars)
            if key in vocab:
                vocab[key] += 1
            else:
                vocab[key] = 1
        return vocab

    # 训练bpe
    def train(self, corpus):
        self.word_counts = self.build_vocab(corpus)
        words = [w for w in self.word_counts if ' ' in w]

        while len(self.word_counts) < self.max_vocab:
            pair_counts = {}
            for word in words:
                count = self.word_counts[word]
                chars = word.split()
                for pair in self.get_pairs(chars):
                    if pair in pair_counts:
                        pair_counts[pair] += count
                    else:
                        pair_counts[pair] = count

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=lambda x: pair_counts[x])
            new_token = ''.join(best_pair)
            self.merge_rules[best_pair] = new_token

            new_vocab = {}
            new_words = []
            for word in words:
                count = self.word_counts[word]
                chars = word.split()
                i = 0
                while i < len(chars) - 1:
                    if (chars[i], chars[i + 1]) == best_pair:
                        chars = chars[:i] + [new_token] + chars[i + 2:]
                    else:
                        i += 1
                new_word = ' '.join(chars)
                if new_word in new_vocab:
                    new_vocab[new_word] += count
                else:
                    new_vocab[new_word] = count
                new_words.append(new_word)

            self.word_counts = new_vocab
            words = new_words

    # 文本转tokens
    def tokenize(self, text):
        words = re.findall(r'\w+|[^\w\s]', text)
        tokens = []
        for word in words:
            chars = list(word) + ['</w>']
            i = 0
            while i < len(chars) - 1:
                pair = (chars[i], chars[i + 1])
                if pair in self.merge_rules:
                    chars = chars[:i] + [self.merge_rules[pair]] + chars[i + 2:]
                    i = max(0, i - 1)
                else:
                    i += 1
            tokens.extend(chars)
        return tokens


# 加载语料
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        return re.findall(r'\w+|[^\w\s]', text)


# 主程序
if __name__ == "__main__":
    corpus = load_text('./corpus.txt')
    print(f"加载完成，共 {len(corpus)} 个词")

    bpe = BPE(5000)
    print("开始训练...")
    bpe.train(corpus)
    print(f"训练完成，词表大小: {len(bpe.word_counts)}")

    test = "韩立望着手中的小瓶，体内的灵力在运转。"
    print("\n序列化结果:")
    print(bpe.tokenize(test))
