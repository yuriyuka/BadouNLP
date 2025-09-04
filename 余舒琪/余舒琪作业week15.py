# 用bpe完成词表构建和文本的序列化
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
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

def build_merges(ids, vocab_size):
    num_merges = vocab_size - 256
    length = len(ids)
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=lambda k: stats[k])
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    print(f"token length: {length}")
    print(f"ids length: {len(ids)}")
    print(f"compression ratio: {length} / {len(ids):.2f}X")
    return merges

def encode(text, merges):
    tokens = text.encode("utf-8")
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda k: merges.get(k, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

def build_vocab(merges):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

def decode(ids, vocab):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

def main(text, to_tokenize_text):
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    merges = build_merges(tokens, 300)
    vocab = build_vocab(merges)
    ids = encode(to_tokenize_text, merges)
    print(ids)
    text1 = decode(ids, vocab)
    print(text1 == to_tokenize_text)

if __name__ == "__main__":
    with open("try/hlm1.txt", "r", encoding="utf-8") as f:
        text = f.read()
    to_tokenize_text = "早知她来，我就不来了。\n吾观颜良，如插标卖首耳。"
    main(text, to_tokenize_text)


