import sys

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train_bpe(text, vocab_size=276):
    tokens = list(text.encode("utf-8"))
    ids = tokens.copy()
    merges = {}
    num_merges = vocab_size - 256

    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    return merges

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def decode(ids, merges):
    # 反向映射：token -> (left, right)
    reverse_merges = {v: k for k, v in merges.items()}

    # 不断地替换合成 token 为子 token
    while True:
        updated = False
        new_ids = []
        for id in ids:
            if id in reverse_merges:
                new_ids.extend(reverse_merges[id])
                updated = True
            else:
                new_ids.append(id)
        ids = new_ids
        if not updated:
            break

    try:
        return bytes(ids).decode("utf-8")
    except Exception:
        return "[Decode Error]"


def read_text_input():
    print("输入模式：")
    print("1 - 从文件读取文本")
    print("2 - 输入一句话")
    choice = input("请选择输入模式：")

    if choice == "1":
        path = input("请输入文件路径：")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif choice == "2":
        return input("请输入一句话：")
    else:
        print("输入无效")
        sys.exit(1)

def main():
    text = read_text_input()
    print("\n原始文本：", text)

    print("\n训练BPE...")
    merges = train_bpe(text)
    print("训练完成！")

    print("\n对文本进行BPE编码...")
    tokens = encode(text, merges)
    print("编码结果：", tokens)

    print("\n尝试还原原始文本（仅限非合并部分）...")
    restored = decode(tokens, merges)
    print("还原结果：", restored)

if __name__ == "__main__":
    main()
