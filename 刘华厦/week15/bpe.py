# utf8编码用最多4个十六进制数来表示一个unicode字符，每个数字在0~256之间
encoder = list("你好".encode("utf-8"))
print(encoder)

encoder = list("ab".encode("utf-8"))
print(encoder)

# text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
text = "背景故事：也许有人会问：“这个世界是如何形成的？”在所有现存世界中，为什么这个世界的属性如此奇特，如此多样化，其中的生物，文化和传说更是数不胜数呢？“答案，”有人低语道，“就在于巨神们。”"
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
print(tokens)


# 按照bpe的思想，统计每个二元组出现的次数
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


stats = get_stats(tokens)
stats_tokens = sorted(((v, k) for k, v in stats.items()), reverse=True)
print(stats_tokens)


# 将ids中的pair元组替换成idx
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("token合并后的结果：")
print(ids)
print("合并的二元组：")
for k, v in merges.items():
    print(f"{k} => {v}")


def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print("字符编码结果：")
encode_text = encode("背景故事：A programmer's Introduction to Unicode")
print(encode_text)

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
print(vocab)

def decode(ids):
    print(ids)
    tokens = b"".join(vocab[idx] for idx in ids)
    print(tokens)
    text = tokens.decode("utf-8", errors="replace")
    return text

decode_text = decode(encode_text)
print("字符解码结果：")
print(decode_text)
