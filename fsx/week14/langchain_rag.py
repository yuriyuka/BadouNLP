import os
from typing import List
import dashscope
from http import HTTPStatus
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from tabulate import tabulate


class QwenEmbeddings(Embeddings):
    """自定义嵌入模型类，使用DashScope的多模态嵌入API"""

    def __init__(self):
        if not os.environ.get("QWEN"):
            raise ValueError("请设置环境变量 QWEN 为您的DashScope API Key")

    def _query_to_vector(self, text: str) -> List[float]:
        """调用DashScope多模态嵌入API"""
        input_data = [{'text': text}]
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=input_data,
            api_key=os.environ.get("QWEN")
        )

        if resp.status_code == HTTPStatus.OK:
            vector = np.array(resp.output["embeddings"][0]["embedding"])
            return vector.tolist()
        else:
            raise Exception(f"调用DashScope API失败: {resp.message}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return [self._query_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        return self._query_to_vector(text)


def main():
    # 初始化自定义嵌入模型
    embeddings = QwenEmbeddings()

    # 加载文档
    loader = DirectoryLoader("./hero", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # 创建向量存储
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db_langchain"
    )

    # 初始化DashScope LLM
    llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.environ.get("QWEN")
    )

    # 创建RAG链
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 执行 RAG 查询
    rag_result = qa({"query": "请介绍龙骑士"})
    rag_text = rag_result["result"]

    # 直接 LLM 调用
    direct_response = llm.invoke("请介绍龙骑士")
    direct_text = direct_response.content

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


if __name__ == "__main__":
    main()
