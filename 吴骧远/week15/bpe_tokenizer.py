with open('input.txt','r') as f:
    text = f.read()

tokens = text.encode('utf-8')
tokens = list(map(int, tokens))
print("length:", len(text))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(sorted(((v,k) for k,v in stats.items()), reverse=True))

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i+1 < len(ids) and ids[i:i+2] == list(pair):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 276  # 词表大小可调整
num_merges = vocab_size - 256
merges = {}  # 存储结果

# BPE
ids = tokens[:]
for i in range(num_merges):
    stats = get_stats(ids)
    if not stats:
        break
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} -> {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

# 构建词表
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

# 编码
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda pair: merges.get(pair, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

# 解码
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

test_text = "hello world!!dagkfsvjs #$13bji 213pkt%n64 65!@!vq s你ih撒罗尼的!"
encoded = encode(test_text)
decoded = decode(encoded)
print(f"Original: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
