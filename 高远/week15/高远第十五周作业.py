corpus = [
    "我爱自然语言处理",
    "我非常爱自然语言",
    "自然语言处理很有趣",
    "爱学习，自然更有趣"
]

# 1) 训练
from bpe import BPETokenizer

tokenizer = BPETokenizer(vocab_size=200, min_pair_freq=2,
                         special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"])
tokenizer.train(corpus)

# 2) 序列化（encode）
ids = tokenizer.encode("我爱自然语言处理很有趣", add_special_tokens=True)
print(ids)

# 3) 反序列化（decode）
text = tokenizer.decode(ids)
print(text)

# 4) 批量序列化并对齐
batch = ["自然语言处理", "我爱自然语言", "很有趣"]
padded, attn_mask = tokenizer.encode_batch(batch, add_special_tokens=True, max_len=16, pad_to_max=True)
print(padded)      # shape: [B, T]
print(attn_mask)   # 1=有效, 0=padding

# 5) 保存/加载
tokenizer.save("bpe_vocab.json", "bpe_merges.txt")
loaded = BPETokenizer.load("bpe_vocab.json","bpe_merges.txt")
print(loaded.encode("自然语言"))
