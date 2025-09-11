from collections import Counter, defaultdict

class BPEUTF8:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = {}
        self.reverse_merges = {}

    def get_stats(self, corpus):
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        new_corpus = {}
        bigram = pair
        new_symbol = bigram[0] + bigram[1]
        for word, freq in corpus.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == bigram:
                    new_word.append(new_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus[tuple(new_word)] = freq
        return new_corpus

    def fit(self, texts):
        # 将文本转为 UTF-8 字节 token
        corpus = defaultdict(int)
        for text in texts:
            tokens = tuple([chr(b) for b in text.encode("utf-8")])
            corpus[tokens] += 1

        for i in range(self.num_merges):
            pairs = self.get_stats(corpus)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.merges[best] = i
            self.reverse_merges[best[0] + best[1]] = best

    def encode(self, text):
        tokens = [chr(b) for b in text.encode("utf-8")]
        pairs = True
        while pairs:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]
            if not mergeable:
                break
            best = min(mergeable, key=lambda x: x[1])[0]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, tokens):
        split_tokens = []
        for token in tokens:
            stack = [token]
            while stack:
                t = stack.pop()
                if t in self.reverse_merges:
                    a, b = self.reverse_merges[t]
                    stack.append(b)
                    stack.append(a)
                else:
                    split_tokens.append(t)
        byte_array = bytes([ord(t) for t in split_tokens])
        return byte_array.decode("utf-8", errors="replace")

if __name__ == "__main__":
    text_data = [
        """In the age of artificial intelligence, large language models are changing
        the way people interact with technology. From answering questions to
        writing essays, translating languages, and even generating code, these
        models are becoming powerful tools in daily life.

        在人工智能的时代，大型语言模型正在改变人们与技术互动的方式。
        从回答问题到写作、翻译，甚至生成代码，这些模型正在成为日常生活中的强大工具。

        Hello world! 你好，世界！Python 是一门非常流行的编程语言，
        它的简洁与强大使得开发者能够快速构建应用程序。"""
    ]

    bpe = BPEUTF8(num_merges=200)
    bpe.fit(text_data)

    encoded = bpe.encode("languages")
    print("Encoded:", encoded)

    decoded = bpe.decode(encoded)
    print("Decoded:", decoded) 
