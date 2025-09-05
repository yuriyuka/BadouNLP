import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# -------- 配置 --------
data_dir = "Heroes/"           # 文件路径
vocab_size = 5000              # 词表大小
special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]

# -------- 获取文件列表 --------
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
if not files:
    raise ValueError(f"No .txt files found in {data_dir}")

# -------- 初始化 BPE 分词器 --------
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# -------- 定义训练器 --------
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,  #只有出现次数 ≥ min_frequency 的符号对才被合并
    special_tokens=special_tokens
)

# -------- 训练 --------
print(f"Training BPE tokenizer on {len(files)} files...")
tokenizer.train(files, trainer)
print("Training completed!")

# -------- 保存分词器 --------
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer saved to bpe_tokenizer.json")

# -------- 导出词表 --------
tok = Tokenizer.from_file("bpe_tokenizer.json")
vocab = tok.get_vocab()  # dict: {token: id}

with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, idx in sorted(vocab.items(),key=lambda x:x[1]):  
        f.write(f"{idx}\t{token}\n")
print("Vocabulary saved to vocab.txt")

# -------- Encode / Decode 示例 --------
def encode(text):
    """文本 -> token ids"""
    return tok.encode(text).ids

def decode(ids):
    """token ids -> 文本"""
    return tok.decode(ids)

text = "我爱自然语言处理"
ids = encode(text)
print("文本:", text)
print("编码ID:", ids)
print("解码文本:", decode(ids))
