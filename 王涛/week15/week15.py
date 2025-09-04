from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = set()      # 最终词表
        self.merges = []        # 合并规则（有顺序）

    def get_stats(self, tokens):
        """统计相邻符号对的频率"""
        counts = defaultdict(int)
        for token in tokens:
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
        return counts

    def merge_pair(self, pair, tokens):
        """合并最频繁的符号对"""
        new_tokens = []
        bigram = ''.join(pair)
        for token in tokens:
            i = 0
            new_token = []
            while i < len(token):
                if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                    new_token.append(bigram)  # 合并成一个新符号
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_tokens.append(new_token)
        return new_tokens

    def train(self, text):
        """训练 BPE 模型"""
        tokens = [list(word) + ["</w>"] for word in text.split()]
        self.vocab = set(sym for token in tokens for sym in token)
        print(tokens)
        for step in range(1, self.vocab_size + 1):
            # 打印当前词表
            current_vocab = set(sym for token in tokens for sym in token)
            print(f"\n=== 第 {step} 轮 ===")
            print("当前词表:", current_vocab)

            # 统计所有 pair
            pairs = self.get_stats(tokens)
            if not pairs:
                break
            print("当前所有 pair 及频次:")
            for k, v in pairs.items():
                print(f"  {k}: {v}")

            # 选出频率最高的 pair
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]
            print(f"选中合并: {best_pair} (频次 {freq})")

            # 执行合并
            tokens = self.merge_pair(best_pair, tokens)
            # 加入合并规则
            self.merges.append(best_pair)
            # 加入词表
            self.vocab.add(''.join(best_pair))
            # 打印合并后语料
            print("合并后语料:")
            for token in tokens:
                print(" ", token)

        print("\n训练完成，最终词表大小:", len(self.vocab))
        print("最终词表:", self.vocab)

    def encode(self, text):
        """将文本编码为 BPE token"""
        tokens = [list(word) + ["</w>"] for word in text.split()]
        for pair in self.merges:
            tokens = self.merge_pair(pair, tokens)
        return tokens

    def decode(self, encoded_tokens):
        """将 BPE token 解码为原始文本"""
        words = []
        for token in encoded_tokens:
            word = "".join(token).replace("</w>", "")
            words.append(word)
        return " ".join(words)


# 使用示例
if __name__ == "__main__":
    text = "low lowest new wider"
    bpe = BPETokenizer(vocab_size=10)
    bpe.train(text)
    print("---------------\n",bpe.merges)
    encoded = bpe.encode("lownew new")
    print("\nEncoded:", encoded)

    decoded = bpe.decode(encoded)
    print("Decoded:", decoded)
