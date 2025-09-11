#按照bpe的思想，我们统计每个2元组出现次数


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
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

def vocab_build(path):
  with open(path, 'r', encoding='utf-8') as file:
    tokens = file.read()
  # ---
  tokens = tokens.encode("utf-8")
  tokens = list(map(int, tokens))
  vocab_size = 512  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
  num_merges = vocab_size - 256
  ids = list(tokens)  # copy so we don't destroy the original list

  merges = {}  # (int, int) -> int
  for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
  return vocab, merges


def decode(ids, vocab):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text


def encode(text, merges):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

vocab, merges = vocab_build(r"D:\study\ai\录播\week14\week14 大语言模型相关第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes\矮人直升机.txt")
print(merges)
ids = encode("技能1：火箭弹幕, 技能描述：向矮人直升机周围一定范围内的敌方单位齐射导弹。持续%abilityduration%秒。", merges)
print(ids)
print(decode(ids, vocab))