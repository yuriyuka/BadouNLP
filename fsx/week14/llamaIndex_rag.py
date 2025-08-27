from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from typing import List
import dashscope
from http import HTTPStatus
import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.dashscope import DashScope
from tabulate import tabulate


class QwenEmbeddingFunction(BaseEmbedding):
    def __init__(self):
        super().__init__()
        if not os.environ.get("QWEN"):
            raise ValueError("请设置环境变量 QWEN 为您的DashScope API Key")

    def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询文本的嵌入向量"""
        vector = self._query_to_vector(query)
        return vector.tolist() if hasattr(vector, 'tolist') else vector

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        vector = self._query_to_vector(text)
        return vector.tolist() if hasattr(vector, 'tolist') else vector

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本嵌入向量"""
        return [self._get_text_embedding(text) for text in texts]

    # 添加缺失的异步方法
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步获取查询文本的嵌入向量"""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步获取文本的嵌入向量"""
        return self._get_text_embedding(text)

    def _query_to_vector(self, text):
        """调用DashScope多模态嵌入API"""
        input_data = [{'text': text}]
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=input_data,
            api_key=os.environ.get("QWEN")
        )

        if resp.status_code == HTTPStatus.OK:
            vector = np.array(resp.output["embeddings"][0]["embedding"])
            return vector
        else:
            raise Exception(f"调用DashScope API失败: {resp.message}")


# 创建嵌入模型实例
embed_model = QwenEmbeddingFunction()

llm = DashScope(
    model="qwen-turbo",
    api_key=os.environ.get("QWEN")
)

# 加载文档
documents = SimpleDirectoryReader("hero").load_data()

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine(llm)

# 执行 RAG 查询
rag_response = query_engine.query("请介绍龙骑士")
rag_text = str(rag_response)

# 直接 LLM 调用
direct_response = llm.complete("请介绍龙骑士")
direct_text = str(direct_response)

# 创建对比数据
comparison_data = [
    ["响应内容", rag_text, direct_text],
    ["信息来源", "本地文档库", "模型预训练知识"],
    ["响应长度", f"{len(rag_text)} 字符", f"{len(direct_text)} 字符"]
]

# 生成表格
headers = ["对比项目", "RAG 系统", "直接 LLM 调用"]
table = tabulate(comparison_data, headers=headers, tablefmt="grid")
print(table)
