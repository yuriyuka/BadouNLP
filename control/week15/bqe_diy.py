import os

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # 构建二元组，并统计出现的次数
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# 对字典中的值进行排序
def sort_dict(d):
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def get_traning_txt():
    text = ""
    for path in os.listdir("Heroes"):
        file_path = os.path.join("Heroes", path)
        try:
            with open(file_path, encoding='utf-8') as f:
                text += f.read()
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试其他编码
            with open(file_path, encoding='gbk') as f:
                text += f.read()
    return text

def get_traning_data():
    #读取Heroes文件夹下的所有文件，并把所有文件拼接起来
    text = get_traning_txt()
    # text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    encode_text = text.encode("utf-8")
    encode_int_data = list(map(int, encode_text))
    return encode_int_data
def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def decode(ids, merges):
    # 构造 解码字典
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    # 对ids进行解码
    utf8_encode = [vocab[id]  for id in ids]
    tokens = b"".join(utf8_encode)
    text = tokens.decode("utf-8", errors="replace")
    return text

def encode(text,merges):
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf"))) # 由于生成的时候，id 自小向大增长。 那么在编码的时候也要先选最小的，才能保证 构建词表 和 新预料生成词表 都是从小到大的。
    if pair not in merges:
      break
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens


def traning_vocab(vocab_size):
    data = get_traning_data()
    num_merges = vocab_size - 256
    ids = list(data)  # copy so we don't destroy the original list
    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = sort_dict(get_stats(ids))
        top_pair = max(stats, key=lambda x: x[1])
        idx = 256 + i
        print(f"merging {top_pair[0]} into a new token {idx}")
        ids = merge(ids, top_pair[0], idx)
        merges[top_pair[0]] = idx
    return merges


if __name__ == "__main__":
    vocab_size = 288
    merges  = traning_vocab(vocab_size)
    print(merges)
    print(len(merges))
    print(encode("我们都有一个家名字叫中国",merges))
    new_ids = [230, 136, 145, 228, 187, 172, 233, 131, 189, 230, 156, 137, 228, 184, 128, 228, 184, 170, 229, 174, 182, 229, 144, 141, 229, 173, 151, 229, 143, 171, 228, 184, 173, 229, 155, 189]
    result = decode(new_ids, merges)
    print(result)