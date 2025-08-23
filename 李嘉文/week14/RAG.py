"""
对比：不加入 RAG vs 加入 RAG
文档：2024-06-15 发布的一篇“内部新闻”
问题：What did the president say about Justice Breyer?
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------ 1. 准备“最新知识” ------------------
docs = [
    "2024-06-15 WhiteHouse Press Release: "
    "The president thanked Justice Breyer for his distinguished service "
    "and nominated Judge Ketanji Brown Jackson as his successor."
]

# ------------------ 2. 工具初始化 ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
tokenizer.pad_token = tokenizer.eos_token
llm = GPT2LMHeadModel.from_pretrained(llm_name).to(device).eval()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------- 3. 构建向量索引（仅 RAG 用） -------------
emb_dim = embed_model.get_sentence_embedding_dimension()
embeddings = embed_model.encode(docs, convert_to_numpy=True)
index = faiss.IndexFlatIP(emb_dim)   # 内积做余弦相似度
index.add(embeddings.astype("float32"))

# ------------------ 4. 问答函数 ------------------
def ask_no_rag(question: str):
    prompt = f"Q: {question}\nA:"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(
            ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).split("A:")[-1].strip()

def ask_with_rag(question: str, top_k=1):
    q_emb = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    _, top_idx = index.search(q_emb, top_k)
    ctx = "\n".join([docs[i] for i in top_idx[0]])
    prompt = f"Context:\n{ctx}\n\nQ: {question}\nA:"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm.generate(
            ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).split("A:")[-1].strip()

# ------------------ 5. 运行并展示差异 ------------------
question = "What did the president say about Justice Breyer?"
print("=================================")
print("使用RAG")
print(ask_with_rag(question))
print("=================================")
print("不使用RAG")
print(ask_no_rag(question))
print("=================================")
# 不知道怎么添加图片
# 输出结果如下
# 使用RAG
#The president thanked Justice Breyer for his distinguished service and nominated Judge Ketanji Brown Jackson as his successor.
# 不使用RAG
#I’m not aware of any recent statement regarding Justice Breyer.
