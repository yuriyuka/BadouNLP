from collections import Counter, defaultdict

class BPEUTF8:
    def __init__(self, num_merges=1000):
        self.num_merges = num_merges
        self.merges = {}          # 存储合并规则 (pair -> merge_order)
        self.reverse_merges = {}  # 存储反向合并规则 (merged_token -> original_pair)
        self.vocab = set()        # 存储最终的词汇表

    def get_stats(self, corpus):
        """计算语料库中所有相邻字符对的频率"""
        pairs = Counter()
        for word, freq in corpus.items():
            symbols = word
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, corpus):
        """根据最频繁的字符对合并词汇表"""
        new_corpus = {}
        bigram = pair
        new_symbol = bigram[0] + bigram[1]
        # 添加新符号到词汇表
        self.vocab.add(new_symbol)
        
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
        """训练BPE模型"""
        # 初始化词汇表，添加所有单个UTF-8字符
        for text in texts:
            for b in text.encode("utf-8"):
                self.vocab.add(chr(b))
        
        # 将文本转为UTF-8字节token
        corpus = defaultdict(int)
        for text in texts:
            # 将文本编码为UTF-8字节，再转换为字符表示
            tokens = tuple([chr(b) for b in text.encode("utf-8")])
            corpus[tokens] += 1

        print(f"开始BPE训练，初始词汇量: {len(self.vocab)}")
        print(f"准备进行 {self.num_merges} 次合并...")
        
        # 执行合并操作
        for i in range(self.num_merges):
            pairs = self.get_stats(corpus)
            if not pairs:
                print(f"没有更多可合并的字符对，提前结束于第 {i+1} 次合并")
                break
                
            # 找到出现频率最高的字符对
            best = max(pairs, key=pairs.get)
            corpus = self.merge_vocab(best, corpus)
            self.merges[best] = i
            self.reverse_merges[best[0] + best[1]] = best
            
            # 每100次合并打印一次进度
            if (i + 1) % 100 == 0:
                print(f"完成 {i+1}/{self.num_merges} 次合并，当前词汇量: {len(self.vocab)}")

        print(f"BPE训练完成，最终词汇量: {len(self.vocab)}")

    def encode(self, text):
        """将文本编码为BPE子词序列"""
        # 首先将文本转换为UTF-8字节表示
        tokens = [chr(b) for b in text.encode("utf-8")]
        pairs = True
        
        # 应用合并规则
        while pairs:
            # 找到所有可能的相邻对
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            # 筛选出可合并的对
            mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]
            
            if not mergeable:
                break
                
            # 找到最早合并的对（按合并顺序）
            best = min(mergeable, key=lambda x: x[1])[0]
            
            # 执行合并
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
        """将BPE子词序列解码为原始文本"""
        split_tokens = []
        for token in tokens:
            stack = [token]
            while stack:
                t = stack.pop()
                if t in self.reverse_merges:
                    # 如果是合并的符号，拆分后重新处理
                    a, b = self.reverse_merges[t]
                    stack.append(b)
                    stack.append(a)
                else:
                    split_tokens.append(t)
        
        # 将字符转换回字节，再解码为文本
        byte_array = bytes([ord(t) for t in split_tokens])
        return byte_array.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        """返回词汇表大小"""
        return len(self.vocab)

if __name__ == "__main__":
    # 替换后的训练文本数据 - 包含多领域和多语言内容
    text_data = [
        """Quantum computing is revolutionizing information processing by leveraging
        quantum mechanics principles like superposition and entanglement. Unlike
        classical bits, quantum bits (qubits) can exist in multiple states simultaneously,
        enabling exponential speedups for certain problems.
        
        量子计算通过利用叠加和纠缠等量子力学原理，正在彻底改变信息处理方式。
        与经典比特不同，量子比特可以同时存在于多个状态，从而为某些问题提供指数级加速。
        
        機械学習は、アルゴリズムとデータを使用してコンピューターが学習し、
        明示的にプログラムされていないタスクを実行できるようにするAIの一分野です。
        
        数据科学是一个跨学科领域，涉及从结构化和非结构化数据中提取知识和见解。
        它结合了统计学、数据可视化、机器学习和领域专业知识。
        
        Climate change is affecting every country on every continent. It's disrupting
        national economies and affecting lives, costing people, communities and countries
        dearly today and even more tomorrow.
        
        クラウドコンピューティングは、インターネットを介してコンピューターシステム
        リソース（特にデータストレージと計算力）を按需提供するコンピューティングの
        一種で、直接活線接続を必要としません。
        """
    ]

    # 创建并训练BPE模型
    bpe = BPEUTF8(num_merges=300)
    bpe.fit(text_data)

    # 测试编码和解码功能
    test_texts = [
        "quantum computing and machine learning",
        "量子计算与人工智能的结合",
        "機械学習とデータサイエンス",
        "climate change impacts on global economy"
    ]

    for text in test_texts:
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print(f"\n原始文本: {text}")
        print(f"编码结果: {encoded[:10]}...")  # 只显示前10个子词
        print(f"解码结果: {decoded}")
        print(f"编码长度: {len(encoded)}")
