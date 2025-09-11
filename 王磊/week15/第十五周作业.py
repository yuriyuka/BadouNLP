import re
from collections import defaultdict
import os


class BPE:
    def __init__(self, vocab_size=1000):
        """
        初始化BPE编码器

        参数:
            vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size  # 目标词汇表大小
        self.vocab = {}  # 词汇表：token -> id
        self.merges = {}  # 合并规则：字符对 -> 合并顺序
        self.reverse_vocab = {}  # 反向词汇表：id -> token

    def get_stats(self, vocab):
        """
        获取字符对的频率统计

        参数:
            vocab: 当前词汇表统计 {word: frequency}

        返回:
            defaultdict: 字符对频率统计 {(char1, char2): frequency}
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            # 将空格分隔的token序列拆分为列表
            symbols = word.split()
            # 统计所有相邻字符对的出现频率
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        合并指定的字符对

        参数:
            pair: 要合并的字符对 (char1, char2)
            vocab: 当前词汇表统计

        返回:
            dict: 合并后的新词汇表
        """
        # 创建正则表达式模式来匹配要合并的字符对
        # 使用re.escape处理可能包含特殊字符的情况
        bigram_pattern = re.escape(' '.join(pair))
        # 使用正则表达式确保精确匹配整个字符对
        pattern = re.compile(r'(?<!\S)' + bigram_pattern + r'(?!\S)')

        new_vocab = {}
        for word, freq in vocab.items():
            # 替换所有匹配的字符对
            new_word = pattern.sub(''.join(pair), word)
            new_vocab[new_word] = freq

        return new_vocab

    def train(self, file_paths):
        """
        从多个文件训练BPE模型

        参数:
            file_paths: 包含训练语料的文件路径列表

        异常:
            ValueError: 如果语料为空或文件不存在
        """
        # 初始化词汇表统计
        vocab = defaultdict(int)

        # 读取所有文件并构建初始词汇表
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"警告: 文件 {file_path} 不存在，跳过")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # 跳过空行
                            continue

                        # 将文本转换为小写并分割为单词
                        words = line.lower().split()
                        for word in words:
                            if not word:  # 跳过空单词
                                continue

                            # 将单词拆分为字符并在字符间添加空格
                            # 例如: "hello" -> "h e l l o </w>"
                            processed_word = ' '.join(list(word)) + ' </w>'
                            vocab[processed_word] += 1

            except UnicodeDecodeError:
                print(f"警告: 文件 {file_path} 编码错误，跳过")
                continue
            except Exception as e:
                print(f"警告: 读取文件 {file_path} 时出错: {e}，跳过")
                continue

        # 检查语料是否为空
        if not vocab:
            raise ValueError("语料为空，请检查文件路径和内容")

        print(f"训练语料包含 {len(vocab)} 个唯一单词形式")

        # 获取基础字符集（所有出现的字符）
        base_vocab = set()
        for word in vocab:
            for char in word.split():
                base_vocab.add(char)

        print(f"基础字符集大小: {len(base_vocab)}")

        # 计算需要执行的合并次数
        num_merges = self.vocab_size - len(base_vocab)
        if num_merges <= 0:
            print(f"词汇表大小 {self.vocab_size} 小于基础字符数量 {len(base_vocab)}")
            print("使用基础字符集作为词汇表")
            self.vocab = {char: idx for idx, char in enumerate(sorted(base_vocab))}
            self.merges = {}
            self._build_reverse_vocab()
            return

        print(f"需要执行 {num_merges} 次合并操作")

        # 执行合并操作
        merges = {}  # 存储合并规则 {(char1, char2): merge_order}

        for merge_step in range(num_merges):
            # 获取当前字符对统计
            pairs = self.get_stats(vocab)

            # 如果没有可合并的字符对，提前结束
            if not pairs:
                print(f"提前结束: 没有更多可合并的字符对 (第 {merge_step} 步)")
                break

            # 找到最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            # 记录合并规则
            merges[best_pair] = merge_step

            if (merge_step + 1) % 100 == 0:
                print(f"合并步骤 {merge_step + 1}: 合并 {best_pair} (频率: {best_freq})")

            # 在词汇表中应用合并
            vocab = self.merge_vocab(best_pair, vocab)

        # 构建最终词汇表
        print("构建最终词汇表...")

        # 首先添加所有基础字符
        self.vocab = {}
        for idx, char in enumerate(sorted(base_vocab)):
            self.vocab[char] = idx

        current_idx = len(self.vocab)

        # 然后添加所有合并产生的新token
        for pair in merges:
            merged_token = ''.join(pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = current_idx
                current_idx += 1

        # 保存合并规则
        self.merges = merges

        # 构建反向词汇表
        self._build_reverse_vocab()

        print(f"训练完成! 最终词汇表大小: {len(self.vocab)}")
        print(f"执行的合并次数: {len(merges)}")

    def _build_reverse_vocab(self):
        """构建反向词汇表 (id -> token)"""
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode_word(self, word):
        """
        编码单个单词为BPE token序列

        参数:
            word: 要编码的单词

        返回:
            list: BPE token列表
        """
        # 添加结束标记
        tokens = list(word.lower()) + ['</w>']

        # 应用所有合并规则（按合并顺序优先级）
        while len(tokens) > 1:
            # 获取所有可能的字符对
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            # 找到优先级最高的合并对（合并顺序数字最小的）
            best_pair = None
            best_priority = float('inf')  # 数字越小，优先级越高

            for pair in pairs:
                if pair in self.merges:
                    priority = self.merges[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair

            # 如果没有可合并的对，结束循环
            if best_pair is None:
                break

            # 合并最佳字符对
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 检查当前和下一个token是否是要合并的对
                if (i < len(tokens) - 1 and
                        (tokens[i], tokens[i + 1]) == best_pair):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2  # 跳过下一个token
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text):
        """
        编码文本为BPE token ID序列

        参数:
            text: 要编码的文本

        返回:
            tuple: (token_id列表, token字符串列表)
        """
        # 预处理文本：转换为小写
        text = text.lower()
        words = text.split()

        encoded_tokens = []

        for word in words:
            if not word:  # 跳过空单词
                continue

            # 编码单个单词
            word_tokens = self.encode_word(word)
            encoded_tokens.extend(word_tokens)

        # 转换为ID序列
        token_ids = []
        for token in encoded_tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # 处理未知token：使用基础字符或第一个token
                # 这里简化处理，使用0（通常是空格或常见字符）
                token_ids.append(0)

        return token_ids, encoded_tokens

    def decode(self, token_ids):
        """
        从BPE token ID序列解码为文本

        参数:
            token_ids: BPE token ID列表

        返回:
            str: 解码后的文本
        """
        # 将ID转换为token字符串
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                # 处理未知ID：使用空格
                tokens.append(' ')

        # 合并所有token
        text = ''.join(tokens)

        # 处理结束标记：将</w>替换为空格
        text = text.replace('</w>', ' ')

        # 清理多余空格
        text = ' '.join(text.split())

        return text.strip()

    def save_vocab(self, file_path):
        """保存词汇表到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")

    def load_vocab(self, file_path):
        """从文件加载词汇表"""
        self.vocab = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, token = line.strip().split('\t', 1)
                self.vocab[token] = int(idx)
        self._build_reverse_vocab()


# 使用示例和测试函数
def test_bpe():
    """测试BPE编码器"""

    # 创建示例语料文件
    sample_texts = [
        "hello world",
        "this is a test",
        "the quick brown fox jumps over the lazy dog",
        "apple banana cherry date elderberry"
    ]
    # 创建BPE实例
    bpe = BPE(vocab_size=1000)

    # 训练模型
    print("开始训练BPE模型...")

    folder_path = "Heroes"
    filearr =[]
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            filearr.append(os.path.join(folder_path, file_name))
    bpe.train(filearr)
    # 测试编码解码
    test_cases = [
        "上古巨神",
        "主宰",
        "先知",
        "不存在啦啦啦"  # 测试未知词处理
    ]

    print("\n测试编码解码:")
    for test_text in test_cases:
        print(f"\n原始文本: '{test_text}'")

        # 编码
        token_ids, tokens = bpe.encode(test_text)
        print(f"BPE tokens: {tokens}")
        print(f"Token IDs: {token_ids}")

        # 解码
        decoded = bpe.decode(token_ids)
        print(f"解码文本: '{decoded}'")
        print(f"匹配: {test_text.lower() == decoded}")



    print(f"\n最终词汇表大小: {len(bpe.vocab)}")
    print("前20个词汇表项:")
    for token, idx in sorted(bpe.vocab.items(), key=lambda x: x[1])[:20]:
        print(f"  {idx}: '{token}'")


if __name__ == "__main__":
    test_bpe()
