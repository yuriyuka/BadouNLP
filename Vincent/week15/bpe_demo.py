class BPE:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.base_vocab_size = 256  # 初始 byte 级词表大小
        self.merges = {}            # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(self.base_vocab_size)}

    def get_stats(self, ids):
        """统计所有相邻 pair 的出现频率"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_ids(self, ids, pair, idx):
        """将 ids 中的指定 pair 合并为新 token"""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text):
        """基于训练文本学习 merge 规则"""
        tokens = list(text.encode('utf-8'))
        num_merges = self.vocab_size - self.base_vocab_size

        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = self.base_vocab_size + i
            tokens = self.merge_ids(tokens, pair, idx)
            self.merges[pair] = idx

        # 根据 merges 建立 vocab
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        """将文本编码为 token 序列"""
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            tokens = self.merge_ids(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        """将 token 序列解码回文本"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")

    def print_vocab(self):
        for idx, token_bytes in self.vocab.items():
            try:
                token_str = token_bytes.decode("utf-8")
            except UnicodeDecodeError:
                token_str = str(token_bytes)
            print(f"{idx}: {token_str}")


bpe = BPE(vocab_size=500)

# 训练
corpus = "A Programmer’s Introduction to Unicode"
bpe.train(corpus)
bpe.print_vocab()

# 编码与解码
encoded = bpe.encode("A Programmer’s Introduction to Unicode")
decoded = bpe.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
