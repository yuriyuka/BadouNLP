import math
from collections import defaultdict


def get_vocab():
    vocab = {}


def get_text_data():
    with open('content.txt', encoding='utf8', mode='r') as f:
        content = f.read()
        print(f'训练文本元素个数:{len(content)}')
        return list(content.encode('utf-8'))


def add_vocab(text_encode, vocab, vocab_i, vocab_size, len_encode):
    m = defaultdict(int)
    max_cup = ()
    max_v = 0
    for i in range(len_encode - 1):
        b = text_encode[i]
        a = text_encode[i + 1]
        index = b * vocab_size + a
        m[index] += 1
        if m[index] > max_v:
            max_v = m[index]
            max_cup = (b, a)
    b, a = max_cup
    vocab[vocab_i] = [b, a]
    l = 0
    r = 0
    while r < len_encode - 1:
        if text_encode[r] == b and text_encode[r + 1] == a:
            text_encode[l] = vocab_i
            r += 2
        else:
            text_encode[l] = text_encode[r]
            r += 1
        l += 1
    return len_encode - max_v, max_v


def bulid_len_dict(vocab):
    max_word_len = 0
    dict_map = {}
    prefix_vocab = {}
    for i in range(len(vocab)):
        prefix_vocab[i] = vocab[i]
    while True:
        is_break = True
        for i in range(255, len(vocab)):
            if max(prefix_vocab[i]) > 255:
                is_break = False
                new_arr = []
                for v in prefix_vocab[i]:
                    if v > 255:
                        for n in prefix_vocab[v]:
                            new_arr.append(n)
                    else:
                        new_arr.append(v)
                prefix_vocab[i] = new_arr
        if is_break:
            break

    dict_list = [prefix_vocab[k] for k in prefix_vocab]
    for line in dict_list:
        word_len = len(line)
        if word_len not in dict_map:
            dict_map[word_len] = []
        dict_map[word_len].append(line)
        max_word_len = max(max_word_len, word_len)
    return max_word_len, dict_map, prefix_vocab


def list_to_num(l):
    res = 0
    for i in l:
        res = res * 300
        res += i
    return res


def encode(content, max_word_len, dict_map, prefix_vocab):
    reverse_map = {list_to_num(prefix_vocab[k]): k for k in prefix_vocab}
    encode_bytes = list(content.encode('utf-8'))
    print(f'压缩前文本长度:{len(encode_bytes)}')
    res = []
    text_len = len(encode_bytes)
    i = 0
    while i < text_len:
        width = min(text_len - i, max_word_len)
        while width > 0:
            seg = encode_bytes[i:i + width]
            if width in dict_map and seg in dict_map[width]:
                res.append(reverse_map[list_to_num(seg)])
                i += width
                break
            else:
                width -= 1
    print(f'压缩后文本长度:{len(res)}')
    return res


def decode(content, vocab):
    encode_bytes = []
    for i in content:
        for item in vocab[i]:
            encode_bytes.append(item)
    return bytes(encode_bytes).decode('utf-8')
    pass


vocab = {}
for i in range(256):
    vocab[i] = [i]
vocab_size = 300
num_merges = vocab_size - 256
text_encode = get_text_data()
len_encode = len(text_encode)
for i in range(num_merges):
    print(f'第{i + 1}次压缩,压缩前编码大小为:{len_encode}')
    vocab_i = 256 + i
    len_encode, count = add_vocab(text_encode, vocab, vocab_i, vocab_size, len_encode)
    print(
        f'第{i + 1}次压缩,压缩后编码大小为:{len_encode}, 新增二元组:key = {vocab_i}, value = {vocab[vocab_i]}, 压缩元素个数: {count}')

max_word_len, dict_map, prefix_vocab = bulid_len_dict(vocab)
test_encode = encode('但是现在，他以五百年的人生经历，重新审视这段历程，心中却波澜不惊，没有一点恨意。', max_word_len,
                     dict_map, prefix_vocab)
print(decode(test_encode, prefix_vocab))
