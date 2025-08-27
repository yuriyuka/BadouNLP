#生成词表

with open("矮人直升机.txt", "r", encoding="utf-8") as f:
    text = f.read()
    tokens = text.encode("utf-8")
    def get_stats(ids):
        counts = {}
        for pair in zip(ids,ids[1:]):
            counts[pair] = counts.get(pair,0)+1
        return counts
    def merge(ids,pair,idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1]==pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    vocab_size = 300 #超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    ids = list(tokens)
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256+i
        ids = merge(ids,pair,idx)
        merges[pair] = idx
        print(f"merging{pair} into a new token {idx}")

#encoding 把词表转成token序列
    def tokenize(text, merges):
        tokens = list(text.encode("utf-8"))
        for pair, idx in merges.items():
            tokens = merge(tokens,pair,idx)
        return tokens
    final_tokens = tokenize(text,merges)
    print("Token序列:", final_tokens)
    print("序列长度:",len(final_tokens))


#decoding
def decode(ids,merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    tokens = b"".join(vocab[idx] for idx in ids)
    decoded_text = tokens.decode("utf-8",errors = "replace")
    return decoded_text

if __name__ == "__main__":
    try:
        decoded_text = decode(final_tokens, merges)
        print("Decoded Text:", decoded_text)
    except Exception as e:
        print(f"解码操失败：{str(e)}")
