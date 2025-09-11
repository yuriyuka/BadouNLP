import os

file_path = r"D:\nlp516\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
intro = ""

if os.path.isdir(file_path):
    for filename in sorted(os.listdir(file_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(file_path, filename), "r", encoding="utf-8") as file:
                intro += file.read()
else:
    with open(file_path, "r", encoding="utf-8") as file:
        intro = file.read()

tokens = intro.encode("utf-8") 
tokens = list(map(int, tokens)) 

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

vocab_size = 768
num_merges = vocab_size - 256
ids = list(tokens) 

import json
merges = {} 
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

merges_serializable = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
vocab_serializable = {k: list(v) for k, v in vocab.items()}

with open("merges.json", "w", encoding="utf-8") as f:
    json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
print("merges and vocab saved to merges.json and vocab.json")

vocab_chinese = {}
for idx, byte_sequence in vocab.items():
    try:
        chinese_char = byte_sequence.decode("utf-8")
        vocab_chinese[idx] = chinese_char
    except UnicodeDecodeError:
        vocab_chinese[idx] = f"<undecodable_bytes:{list(byte_sequence)}>"
with open("vocab_chinese.json", "w", encoding="utf-8") as f:
    json.dump(vocab_chinese, f, ensure_ascii=False, indent=2)
print("Vocabulary mapping saved to vocab_chinese.json")

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

entlist = encode("将所有连续出现的 pair 替换为新令牌 IDX 。")
print(entlist)

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode(entlist))
print(encode(decode(entlist))==entlist)

# 输出（outputs) 如下：
# tokens length: 289040
# ids length: 116711
# compression ratio: 2.48X
# merges and vocab saved to merges.json and vocab.json
# Vocabulary mapping saved to vocab_chinese.json
# [342, 487, 266, 158, 459, 395, 512, 259, 32, 112, 97, 105, 114, 32, 633, 191, 557, 162, 372, 290, 176, 263, 164, 375, 140, 32, 73, 68, 88, 32, 262]
# 将所有连续出现的 pair 替换为新令牌 IDX 。
# True