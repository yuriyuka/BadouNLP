import  json
import os
import jieba
import numpy as np
from openai import OpenAI
from config import Config
from bm25 import BM25


def call_large_model(prompt):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=Config["api_key"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[

            {"role": "user", "content": prompt},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    return completion.choices[0].message.content

class SimpleRAG:
    def __init__(self,folder_path='/Library/workerspace/python_test/badou2/week14/Heroes'):
        self.load_hero_data(folder_path)
    def load_hero_data(self,folder_path):
        self.hero_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path,file_name),"r",encoding="utf-8") as f:
                    intro = f.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for hero,intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrieve(self,user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores,key = lambda x :x[1], reverse = True)
        hero = sorted_scores[0][0]
        text = self.hero_data[hero]
        return text

    def query(self,user_query):
        print("user_query",user_query)
        print("-------------------------------")
        retrieve_text = self.retrieve(user_query)
        print("-------------------------------")
        prompt =f"请根据以下从数据库中获得的英雄故事和技能介绍，回答用户问题：\n\n英雄故事及技能介绍：\n{retrieve_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答:",response_text)
        print("-------------------------------")
if __name__ =="__main__":
    rag = SimpleRAG()
    user_query = "高射火炮是谁的技能"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))