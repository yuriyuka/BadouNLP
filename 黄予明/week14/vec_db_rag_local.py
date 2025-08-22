# RAG系统 - 使用本地Embedding模型
# 无需API Key，完全离线运行

import os
import numpy as np
import chromadb
from chromadb.config import Settings
from local_embedding import LocalEmbeddingFunction, query_to_vector_local
from bm25 import BM25
import re

print("🏠 启动本地RAG系统...")
print("✅ 无需API Key，完全离线运行!")

# 初始化ChromaDB客户端
client = chromadb.Client(Settings(
    persist_directory="./chroma_db_local",  # 本地存储目录
    is_persistent=True
))

# 清理旧数据（可选）
collections = client.list_collections()
for collection in collections:
    client.delete_collection(name=collection.name)
    print(f"已删除旧集合: {collection.name}")

# 创建使用本地Embedding的集合
collection_name = "rag_local"
print("🔄 正在创建集合并加载本地embedding模型...")

# 选择embedding模型（根据需要调整）
model_configs = {
    "lightweight": "all-MiniLM-L6-v2",          # 轻量级，23MB
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"  # 多语言，266MB
}

# 使用轻量级模型（推荐）
embedding_model = model_configs["multilingual"]

collection = client.create_collection(
    name=collection_name,
    embedding_function=LocalEmbeddingFunction(model_name=embedding_model)
)
print(f"✅ 已创建本地embedding集合: {collection_name}")

# 从Heroes文件夹加载英雄文档
def load_hero_documents():
    """加载Heroes文件夹下的所有英雄文档"""
    heroes_dir = "./Heroes"
    documents = []
    ids = []
    
    if not os.path.exists(heroes_dir):
        print(f"❌ Heroes文件夹不存在: {heroes_dir}")
        return [], []
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(heroes_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"❌ 在{heroes_dir}中没有找到txt文件")
        return [], []
    
    print(f"📁 正在加载 {len(txt_files)} 个英雄文档...")
    
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(heroes_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 确保文件不为空
                    documents.append(content)
                    # 使用文件名（去掉.txt后缀）作为ID
                    hero_name = filename.replace('.txt', '')
                    ids.append(f"hero_{hero_name}")
                    print(f"   ✅ 已加载: {hero_name}")
        except Exception as e:
            print(f"   ❌ 加载失败 {filename}: {e}")
    
    print(f"📊 成功加载 {len(documents)} 个英雄文档")
    return documents, ids

# 加载英雄文档
documents, ids = load_hero_documents()

if not documents:
    print("❌ 没有加载到任何文档，程序退出")
    exit(1)

print("📝 正在添加文档到向量数据库...")
print("   第一次运行可能需要一些时间来处理向量...")

try:
    collection.add(
        documents=documents,
        ids=ids
    )
    print(f"✅ 成功添加 {len(documents)} 个文档")
except Exception as e:
    print(f"❌ 添加文档失败: {e}")
    exit(1)

# -----------------------------
# BM25 索引与混合检索
# -----------------------------

def tokenize(text: str):
    """极简分词：
    - 提取中文、英文、数字片段
    - 中文按单字作为 token（无第三方依赖，兼容性最好）
    """
    tokens = []
    # 英文/数字词
    tokens.extend(re.findall(r"[A-Za-z0-9]+", text.lower()))
    # 中文单字
    chinese_spans = re.findall(r"[\u4e00-\u9fff]+", text)
    for span in chinese_spans:
        tokens.extend(list(span))
    return tokens

# 构建 BM25 语料（使用与 Chroma 相同的 ids 对应）
id_to_doc = {ids[i]: documents[i] for i in range(len(ids))}
bm25_corpus = {ids[i]: tokenize(documents[i]) for i in range(len(documents))}
bm25_index = BM25(bm25_corpus)
print(f"🔧 已构建 BM25 索引，文档数: {len(bm25_corpus)}，平均长度: {bm25_index.avgdl:.1f}")

def search_bm25(query: str, top_k: int = 20):
    query_tokens = tokenize(query)
    scores = bm25_index.get_scores(query_tokens)
    # scores: List[[doc_id, score]]，来自实现
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    return scores_sorted  # [(doc_id, score)]

def search_vector(query: str, top_k: int = 20):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "ids", "distances"]
        )
        # 统一返回 [(doc_id, score)]，将距离转为相似度分数
        vec_ids = results.get("ids", [[]])[0]
        vec_docs = results.get("documents", [[]])[0]
        vec_dists = results.get("distances", [[]])[0]
        pairs = []
        for did, dist in zip(vec_ids, vec_dists):
            # 将较小的距离转为较大的分数；添加一个稳定项防 0
            score = 1.0 / (1e-6 + dist)
            pairs.append((did, score))
        return pairs
    except Exception as e:
        print(f"检索失败(Vector): {e}")
        return []

def rrf_fusion(bm25_list, vec_list, k: int = 60):
    """RRF 融合：
    输入：
      - bm25_list: [(doc_id, score)]，按分数降序
      - vec_list: [(doc_id, score)]，按分数降序
    返回：doc_id -> 融合分数
    """
    rank_map = {}
    # 对 BM25 排名
    for rank, (doc_id, _score) in enumerate(bm25_list, start=1):
        rank_map.setdefault(doc_id, 0.0)
        rank_map[doc_id] += 1.0 / (k + rank)
    # 对向量 排名
    for rank, (doc_id, _score) in enumerate(vec_list, start=1):
        rank_map.setdefault(doc_id, 0.0)
        rank_map[doc_id] += 1.0 / (k + rank)
    return rank_map

def search_hybrid(query: str, top_k: int = 5, bm25_k: int = 30, vec_k: int = 30):
    bm25_res = search_bm25(query, top_k=bm25_k)
    vec_res = search_vector(query, top_k=vec_k)
    fused = rrf_fusion(bm25_res, vec_res)
    # 排序取前 top_k
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # 返回 documents 列表
    docs = [id_to_doc[doc_id] for doc_id, _ in ranked if doc_id in id_to_doc]
    debug = {
        "bm25_top_ids": [d for d, _ in bm25_res[:10]],
        "vec_top_ids": [d for d, _ in vec_res[:10]],
        "fused_top_ids": [d for d, _ in ranked],
    }
    return docs, debug

# 检索相关文档
def search_similar(query, top_k=3):
    """混合检索：BM25 + 向量 + RRF"""
    try:
        docs, debug = search_hybrid(query, top_k=top_k)
        print(f"\n🔎 Hybrid 调试: \n   BM25前10: {debug['bm25_top_ids']}\n   向量前10: {debug['vec_top_ids']}\n   融合Top: {debug['fused_top_ids']}")
        return {"documents": [docs]}
    except Exception as e:
        print(f"检索失败(Hybrid): {e}")
        return None

# 定义检索函数
def retrieve_documents(query, top_k=3):
    """检索相关文档"""
    retrieved_docs = search_similar(query, top_k=top_k)
    if retrieved_docs and retrieved_docs["documents"]:
        return retrieved_docs["documents"][0]
    return []

# 简单的RAG生成函数（无需大模型API）
def rag_retrieve_and_summarize(query):
    """检索相关文档并提供简单总结"""
    retrieved_docs = retrieve_documents(query, top_k=3)
    
    print(f"\n🔍 查询: {query}")
    print("📄 检索到的相关文档:")
    for i, doc in enumerate(retrieved_docs):
        print(f"   {i+1}. {doc}")
    
    # 简单的基于规则的总结
    context = "\n".join(retrieved_docs)
    
    # 计算查询与文档的相关性得分（简单实现）
    if retrieved_docs:
        print(f"\n📊 基于检索的信息总结:")
        print(f"   相关文档数量: {len(retrieved_docs)}")
        print(f"   主要内容关键词: {extract_keywords(context)}")
        print(f"\n💡 建议答案基于以下内容:")
        print(f"   {context[:200]}..." if len(context) > 200 else context)
    
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "keywords": extract_keywords(context) if retrieved_docs else []
    }

def extract_keywords(text):
    """简单的关键词提取"""
    import re
    # 简单的中英文关键词提取
    keywords = []
    # 中文词汇
    chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
    keywords.extend([w for w in chinese_words if len(w) >= 2])
    
    # 英文词汇
    english_words = re.findall(r'\b[A-Za-z]{3,}\b', text)
    keywords.extend(english_words)
    
    # 去重并取前5个
    return list(set(keywords))[:5]


# 对外暴露的接口：根据查询生成给大模型的提示词
def get_rag_prompt(query: str, top_k: int = 3) -> str:
    """
    外部调用入口：
    - 输入 query
    - 先通过 RAG 检索与汇总
    - 返回给大模型的中文提示词（包含上下文资料）
    """
    result = rag_retrieve_and_summarize(query)
    context = result.get("context", "") if isinstance(result, dict) else ""
    prompt = (
        "请基于以下资料回答用户问题。如果资料中未包含答案，请明确说明。\n"
        f"用户问题：{query}\n"
        "资料（可能为多段，按相关性排序）：\n"
        f"{context}\n\n"
        "回答要求：\n"
        "- 仅依据资料作答，不要编造\n"
        "- 如果资料不足以回答，请说明\n"
        "- 中文作答，给出简洁且准确的答案\n"
    )
    return prompt

# 主程序
if __name__ == "__main__":

    
    #  显示系统信息
    print(f"\n📊 系统信息:")
    print(f"   向量数据库: ChromaDB")
    print(f"   嵌入模型: {embedding_model}")
    print(f"   英雄文档数量: {len(documents)}")
    print(f"   存储位置: ./chroma_db_local")
    print(f"   运行模式: 完全离线")
    print(f"   数据来源: Heroes文件夹")
    
    # 简单的单次查询演示
    demo_query = "风行者"
    print(f"\n🎯 演示查询: {demo_query}")
    result = rag_retrieve_and_summarize(demo_query)
    print(result)
