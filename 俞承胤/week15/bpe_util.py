

def get_stats(ids):
    counts = {}
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids,pair,idx):
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

if __name__ == "__main__":
    vocab_size = 300
    num_merges = vocab_size - 256
    text = "关于这个事，我简单说两句，你明白就行，总而言之，这个事呢，现在就是这个情况，具体的呢，大家也都看得到，也得出来说那么几句，可能，你听的不是很明白，但是意思就是那么个意思，不知道的你也不用去猜，这种事情见得多了，我只想说懂得都懂，不懂的我也不多解释，毕竟自己知道就好，细细品吧。你们也别来问我怎么了，利益牵扯太大，说了对你我都没好处，当不知道就行了，其余的我只能说这里面水很深，牵扯到很多东西。详细情况你们自己是很难找的，网上大部分已经删除干净了，所以我只能说懂得都懂。懂的人已经基本都获利上岸什么的了，不懂的人永远不懂，关键懂的人都是自己悟的，你也不知道谁是懂的人也没法请教。"
    print("文本长度：",len(text))
    tokens = list(map(int,text.encode("utf-8")))
    ids = list(tokens)
    print("utf-8编码长度：",len(ids))
    stats = []
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    print("bpe计算后ids的长度：",len(ids))
    print(merges)
    new_stats = list(ids)


    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    print(vocab)
    def decode(ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    print(decode(new_stats))
