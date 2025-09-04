import os


# 基础函数：计算统计信息和合并操作
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# 训练BPE：生成merges规则和vocab字典
def bpe_train(raw_sequences, target_vocab_size=276):
    initial_vocab_size = 256
    num_merges = target_vocab_size - initial_vocab_size

    # 初始化：merges记录合并规则，vocab记录ID到字节的映射
    merges = {}  # {(p1, p2): new_id, ...}
    vocab = {i: bytes([i]) for i in range(initial_vocab_size)}

    # 对所有序列的合并规则进行训练（这里简化为用第一个序列训练）
    current = raw_sequences[0].copy() if raw_sequences else []
    for step in range(num_merges):
        stats = get_stats(current)
        if not stats:
            break  # 序列过短，无法继续合并
        # 选择频率最高的对进行合并
        top_pair = max(stats, key=stats.get)
        new_id = initial_vocab_size + step
        # 记录合并规则和vocab
        merges[top_pair] = new_id
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        # 更新当前序列
        current = merge(current, top_pair, new_id)

    return merges, vocab


# 解码函数：ID序列 -> 字符串
def decode(ids, vocab):
    bytes_data = b"".join(vocab[idx] for idx in ids)
    return bytes_data.decode("utf-8", errors="replace")


# 编码函数：字符串 -> ID序列
def encode(text, merges):
    # 1. 先转为原始字节token
    tokens = list(text.encode("utf-8"))
    # 2. 根据merges规则合并
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 找到merges中存在的、优先级最高（最早合并）的对
        # 优先级：训练时越早合并的对，优先级越高
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # 没有可合并的对了
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


# 主程序
if __name__ == '__main__':
    # 1. 准备数据（加载训练文本）
    path = "Heroes"
    raw_data = []
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            with open(os.path.join(path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                raw_tokens = list(map(int, text.encode("utf-8")))
                raw_data.append(raw_tokens)
    if not raw_data:
        print("请在Heroes文件夹中放入文本文件")
        exit()

    # 2. 训练BPE，得到合并规则和vocab
    merges, vocab = bpe_train(raw_data, target_vocab_size=276)
    print(f"训练完成，共记录 {len(merges)} 条合并规则")

    # 3. 测试编码：将新字符串转为ID序列
    test_text = "背景故事：艾欧存在于所有地方和世间万物之内。敌人们诋毁它为灭绝者，学者们则尊崇它为闪动的神圣之眼，艾欧同时在所有位面存在着，它们可以在任何时候将自己身体的一小部分转化为物理存在。"
    encoded_ids = encode(test_text, merges)
    print(f"\n编码结果（{test_text}）：")
    print(encoded_ids)

    # 4. 测试解码：将ID序列转回字符串
    decoded_text = decode(encoded_ids, vocab)
    print(f"\n解码结果：{decoded_text}")
    print(f"编码解码一致性：{test_text == decoded_text}")  # 理想情况下应为True
