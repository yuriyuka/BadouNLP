import os
import re


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


def deal_text():
    all_text = ""
    with os.scandir("./Heroes") as entries:
        for entry in entries:
            if entry.is_file():
                with open(entry, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = re.sub(r'<br\s*/?>', '', text)
                    clear_text = re.sub(r'\s+', '', text)
                    all_text += clear_text

    tokens = all_text.encode("utf-8")
    tokens = list(map(int, tokens))

    vocab_size = 276
    num_merges = vocab_size - 256

    merges = {}
    for i in range(num_merges):
        stats = get_stats(tokens)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        tokens = merge(tokens, pair, idx)
        merges[pair] = idx

    print("merges:", merges)
    return merges


def encode(merges, text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


def decode(merges, ids):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text



if __name__ == "__main__":

    merges = deal_text()

    text = "的风景和尼克等会捏客人你快回家呢人空间客人很快就理解呢让他看见还能卡特进入"
    tokens = encode(merges, text)

    print("tokens:", tokens)
    # print(len(text))
    # print(len(tokens))

    decode_text = decode(merges, tokens)
    print(decode_text)
