import json
import os
import jieba
import numpy as np
from zai import ZhipuAiClient
from bm25 import BM25

'''
基于RAG来介绍Dota2英雄故事和技能
用bm25做召回
同样以来智谱的api作为我们的大模型
https://docs.bigmodel.cn/cn/guide/start/model-overview
'''
zhipuApiKey = "4102310e20104cc3b0275e005ffb6398.PnDi2MdmYSz5gWPO"


# 智谱的api作为我们的大模型
def call_large_model(prompt):
    client = ZhipuAiClient(api_key=zhipuApiKey)  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4.5",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text


class SimpleRAG:
    def __init__(self, folder_path="data"):
        self.load_hero_data(folder_path)

    def load_hero_data(self, folder_path):
        self.hero_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for hero, intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        hero = sorted_scores[0][0]
        text = self.hero_data[hero]
        return text

    def query(self, user_query):
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以下从数据库中获得的信息，回答用户问题：\n\n信息：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")


if __name__ == "__main__":
    rag = SimpleRAG()
    user_query = "工作注意事项有哪些？请列出核心5点。"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))
