import os
dir_path = os.path.dirname(os.path.abspath(__file__))
folder = dir_path + '/Heroes'
def read_tokens_in_folder(folder):
    content = ""
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content += f.read()
                # print(f'--- {filename} ---')
    print(content)
    tokens = content.encode("utf-8") # raw bytes
    tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
    return tokens


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

# ---
vocab_size = 2760 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
num_merges = vocab_size - 256
ids = read_tokens_in_folder(folder)
print("initial ids length:", len(ids))
# ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

# print("merges:", merges)
print("final ids length:", len(ids))
