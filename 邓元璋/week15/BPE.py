import re
from collections import defaultdict


class BPEChineseProcessor:
    def __init__(self, vocab_size=1000):
        """初始化BPE处理器

        Args:
            vocab_size: 目标词表大小
        """
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}  # 特殊标记
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.merges = {}  # 存储合并规则 (pair -> merged)
        self.vocab = set(self.word2idx.keys())  # 词表集合

    def get_pairs(self, word):
        """从词中提取相邻字符对"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def count_pairs(self, corpus):
        """统计语料中所有字符对的频率"""
        pair_counts = defaultdict(int)
        for word, freq in corpus.items():
            pairs = self.get_pairs(word.split())
            for pair in pairs:
                pair_counts[pair] += freq
        return pair_counts

    def merge_pair(self, corpus, pair):
        """将最频繁的字符对合并"""
        merged_token = ''.join(pair)
        new_corpus = {}
        pattern = re.escape(' '.join(pair))
        regex = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')

        for word, freq in corpus.items():
            new_word = regex.sub(merged_token, word)
            new_corpus[new_word] = freq

        return new_corpus, merged_token

    def preprocess_text(self, text):
        """预处理中文文本，每个字用空格分隔"""
        # 简单处理，实际应用中可能需要更复杂的预处理
        # 这里我们保留标点符号作为独立符号
        return ' '.join(list(text))

    def build_vocab(self, texts):
        """从文本列表构建BPE词表"""
        # 预处理文本并统计词频
        corpus = defaultdict(int)
        for text in texts:
            processed = self.preprocess_text(text)
            corpus[processed] += 1

        # 初始化词汇 - 所有单个字符
        for word in corpus.keys():
            for char in word.split():
                if char not in self.vocab:
                    self.vocab.add(char)

        # 初始词表大小（包含特殊标记）
        initial_size = len(self.vocab) + 2  # +2是因为已经有了<PAD>和<UNK>
        print(f"初始词表大小: {initial_size}")

        # 需要执行的合并次数
        num_merges = self.vocab_size - initial_size
        if num_merges <= 0:
            print("目标词表大小小于等于初始词表大小，无需合并")
            num_merges = 0

        # 执行BPE合并
        for i in range(num_merges):
            # 计算字符对频率
            pair_counts = self.count_pairs(corpus)
            if not pair_counts:
                break  # 没有更多可合并的对

            # 找到最频繁的字符对
            best_pair = max(pair_counts, key=pair_counts.get)

            # 合并字符对
            corpus, merged_token = self.merge_pair(corpus, best_pair)

            # 记录合并规则
            self.merges[best_pair] = merged_token

            # 将新合并的词添加到词表
            self.vocab.add(merged_token)

            # 打印进度
            if i % 100 == 0 or i == num_merges - 1:
                print(f"完成 {i + 1}/{num_merges} 次合并，当前词表大小: {initial_size + i + 1}")

        # 构建最终的word2idx和idx2word映射
        current_idx = len(self.word2idx)
        for token in self.vocab:
            if token not in self.word2idx:
                self.word2idx[token] = current_idx
                self.idx2word[current_idx] = token
                current_idx += 1

        print(f"BPE词表构建完成，最终词表大小: {len(self.word2idx)}")

    def encode(self, text, max_length=None):
        """将文本编码为BPE子词序列的索引"""
        # 预处理文本
        processed = self.preprocess_text(text)
        tokens = processed.split()

        # 应用合并规则
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                # 合并这对字符
                merged = self.merges[pair]
                tokens = tokens[:i] + [merged] + tokens[i + 2:]
                # 回退一步，检查是否可以继续合并
                if i > 0:
                    i -= 1
                else:
                    i = 0
            else:
                i += 1

        # 转换为索引
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

        # 处理长度
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices += [self.word2idx['<PAD>']] * (max_length - len(indices))

        return indices, tokens  # 返回索引和子词序列

    def decode(self, indices):
        """将索引序列解码为文本"""
        # 过滤填充标记
        indices = [idx for idx in indices if idx != self.word2idx['<PAD>']]
        # 转换为子词
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        # 拼接成文本
        return ''.join(tokens)


# 验证过程
if __name__ == "__main__":
    # 示例中文文本
    texts = [
        "自然语言处理是人工智能的一个重要分支。",
        "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
        "中文处理有其特殊性，因为中文没有明显的词边界。",
        "分词是中文自然语言处理的基础步骤之一。",
        "深度学习在自然语言处理领域取得了显著进展。",
        "词向量是表示词语语义的重要方法。",
        "注意力机制极大地提升了自然语言处理模型的性能。"
    ]

    # 创建BPE处理器实例，设置目标词表大小
    bpe_processor = BPEChineseProcessor(vocab_size=300)

    # 构建词表
    bpe_processor.build_vocab(texts)

    # 打印部分词表
    print("\n部分BPE词表:")
    for i, (word, idx) in enumerate(list(bpe_processor.word2idx.items())[:15]):
        print(f"{word}: {idx}")

    # 选择一个文本进行编码解码测试
    test_text = "中文分词是自然语言处理的重要步骤。"
    print(f"\n原始文本: {test_text}")

    # 编码
    encoded_indices, tokens = bpe_processor.encode(test_text)
    print(f"BPE子词: {tokens}")
    print(f"编码结果: {encoded_indices}")

    # 解码
    decoded_text = bpe_processor.decode(encoded_indices)
    print(f"解码结果: {decoded_text}")

    # 验证是否一致
    if decoded_text == test_text:
        print("验证成功: 解码结果与原始文本一致!")
    else:
        print("验证失败: 解码结果与原始文本不一致!")

    # 测试包含未见过的文本
    new_text = "机器学习与深度学习是人工智能领域的重要技术。"
    print(f"\n新文本: {new_text}")

    encoded_new, tokens_new = bpe_processor.encode(new_text)
    print(f"BPE子词: {tokens_new}")
    print(f"编码结果: {encoded_new}")

    decoded_new = bpe_processor.decode(encoded_new)
    print(f"解码结果: {decoded_new}")
