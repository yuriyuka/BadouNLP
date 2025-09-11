import copy
from collections import defaultdict

class BBPE:
    def __init__(self, train_path, max_vocab_size=300):
        # 读取为 bytes 行 -> 每行为 list of token ids (初始为 0..255)
        with open(train_path, "rb") as f:
            # 每一行转为 list(int)（去掉换行）
            self.train_data = [list(line.strip()) for line in f]

        self.max_vocab_size = max_vocab_size
        # 基础单字节词表（id 与原始 byte tuple 的双向映射）
        self.id_to_token = {i: (i,) for i in range(256)}
        self.token_to_id = { (i,): i for i in range(256) }
        self.next_id = 256

        # 用于训练时的可修改副本（token id 序列）
        self.train_data_modified = copy.deepcopy(self.train_data)

        # 最大可新增 token 数量
        self.max_generate_vocab_size = self.max_vocab_size - 256
        self.generated = 0

        # 当前最长 token（以原始 byte sequence 长度计）
        self.max_token_len = 1

        # 训练
        self.train()

    def get_pair_frequencies(self):
        """返回当前 train_data_modified 中所有相邻 token id 对的频率 dict"""
        freq = defaultdict(int)
        for line in self.train_data_modified:
            for i in range(len(line) - 1):
                pair = (line[i], line[i+1])
                freq[pair] += 1
        return freq

    def _replace_pair_in_line(self, line, pair_to_merge, new_id):
        """line: list of token ids. pair_to_merge: tuple(token_id_a, token_id_b). 将非重叠地替换所有该 pair -> new_id"""
        a, b = pair_to_merge
        res = []
        i = 0
        L = len(line)
        while i < L:
            if i < L - 1 and line[i] == a and line[i+1] == b:
                res.append(new_id)
                i += 2
            else:
                res.append(line[i])
                i += 1
        return res

    def train(self):
        """简单但正确的 BPE：每次选当前最频繁的相邻 pair 合并，合并后重算所有 pair 频率（可优化）"""
        while self.generated < self.max_generate_vocab_size:
            freq = self.get_pair_frequencies()
            if not freq:
                break
            # 选出现次数最多的 pair
            best_pair, best_count = max(freq.items(), key=lambda kv: kv[1])
            if best_count <= 0:
                break

            # 将 best_pair 的两个 token id -> 展开为原始 byte 序列并拼接，作为新 token 的原始序列
            left_id, right_id = best_pair
            left_seq = self.id_to_token[left_id]
            right_seq = self.id_to_token[right_id]
            new_seq = left_seq + right_seq  # tuple 拼接，代表原始 bytes 序列

            # 登记到词表（token_to_id 使用原始 byte tuple 作为 key）
            new_id = self.next_id
            self.next_id += 1
            self.id_to_token[new_id] = new_seq
            self.token_to_id[new_seq] = new_id
            self.generated += 1
            # 更新最长 token 长度
            if len(new_seq) > self.max_token_len:
                self.max_token_len = len(new_seq)

            # 对语料做替换（基于 token ids）
            for idx, line in enumerate(self.train_data_modified):
                self.train_data_modified[idx] = self._replace_pair_in_line(line, best_pair, new_id)

            # 继续下一轮（我们在循环开始会重新统计频次）
        # 训练结束

    def encode(self, text_bytes: bytes):
        """把 bytes 编码为 token id 序列。
        使用最长前缀匹配：从当前位置贪心匹配词表中最长的原始 byte tuple。
        """
        # 把输入转为原始 bytes 的 int 列表
        data = list(text_bytes)
        i = 0
        out = []
        N = len(data)
        # 为了匹配，我们要匹配原始 byte tuple，所以在每一步尝试从最长到最短匹配
        while i < N:
            matched = False
            # 尝试窗口最大为 max_token_len，但不要超过剩余长度
            max_k = min(self.max_token_len, N - i)
            # 从长的尝试到短的（贪心最长匹配）
            for k in range(max_k, 0, -1):
                candidate = tuple(data[i:i+k])
                if candidate in self.token_to_id:
                    out.append(self.token_to_id[candidate])
                    i += k
                    matched = True
                    break
            if not matched:
                # 如果连单字都匹配不上（理论上不该发生），把单字作为 fallback
                out.append(self.token_to_id[(data[i],)])
                i += 1
        return out

    def decode(self, tokens):
        """把 token id 列表解码回 bytes"""
        parts = []
        for tid in tokens:
            seq = self.id_to_token.get(tid)
            if seq is None:
                raise ValueError(f"unknown token id {tid}")
            parts.extend(seq)
        return bytes(parts)
    

if __name__ == "__main__":
    bbpe = BBPE("train.txt", max_vocab_size=300)
    s = b"hello world"
    encoded = bbpe.encode(s)
    print("encoded:", encoded)
    decoded = bbpe.decode(encoded)
    print("decoded == s:", decoded == s)