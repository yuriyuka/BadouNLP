text = ('''
 BPE 最初是一种数据压缩算法，通过迭代合并数据中最频繁出现的字节对（Byte Pair），逐步构建一个编码表，将高频字节对替换为一个新的符号，从而减少数据中的重复模式，达到压缩目的。其核心逻辑是：通过统计数据中相邻符号的频率，不断合并高频符号对，生成更复杂的新符号，最终将原始数据转换为符号序列，减少数据冗余。
 A Programmer’s Introduction to Unicode March 3, 2017 · Coding · 22 Comments  Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺\u200c🇳\u200c🇮\u200c🇨\u200c🇴\u200c🇩\u200c🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.  A few months ago, I got interested in Unicode and decided to spend some time learning more about it in detail.
''')

tokens = text.encode("utf-8")
tokens = list(map(int,tokens))

def get_stats(ids):
    counts ={}
    for pair in zip(ids,ids[1:]):
        counts [pair] = counts.get(pair,0) + 1
    return counts
def merge(ids,pair,idx):
    newids =[]
    i = 0
    while i < len(ids):
        if i< len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i +=2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 276
num_merges = vocab_size -256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats,key = stats.get)
    idx = 256 + i
    print(f"mergeing {pair} into a new token {idx}")
    ids = merge(ids,pair,idx)
    merges[pair] = idx

print("tokens length:",len(tokens))
print("ids length:",len(ids))
print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

print("=====================================================")

vocab = {idx: bytes([idx])for idx in  range(256)}
for(p0,p1) ,idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8",errors = "replace")
    return text

print("decode 示例:",decode(
[65, 32, 80, 114, 111, 103, 114, 97, 109, 109, 260, 263, 153, 258, 73, 110, 116, 114, 111, 100, 117, 99, 116, 105,
     111, 110, 32, 116, 111, 32, 85, 110, 105, 271, 101,]
))
print("=====================================================")
print(merges)
print("=====================================================")
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens)>=2:
        stats = get_stats(tokens)
        pair = min(stats,key =lambda p:merges.get(p,float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens,pair,idx)
    return tokens

str ='等到黑夜反面之后 会是新的白昼 等到海啸退去之后 只是潮起潮落'
print("encode 示例:",encode("等到黑夜反面之后 会是新的白昼 等到海啸退去之后 只是潮起潮落"))

for i in str:
    print(i,encode(i))
print(str == decode(encode(str)))
