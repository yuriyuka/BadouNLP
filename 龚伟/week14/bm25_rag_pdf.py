import PyPDF2
import os
import jieba
from config import Config
from openai import OpenAI
from bm25 import BM25

def call_large_model(prompt):
    client = OpenAI(
        api_key= Config["api_key"],
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = "qwen-plus",
        messages = [
            {"role": "user", "content": prompt}
        ],
    )
    return completion.choices[0].message.content

class SimpleRAG:
    def __init__(self,folder_path='/Library/workerspace/python_test/badou2/week14/pdf'):
        self.load_data(folder_path)

    def load_data(self,path):
        self.hero_data = {}
        for file_name in os.listdir(path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(path,file_name)
                try:
                    with open(file_path,"rb" ) as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        intro =""
                        for page in pdf_reader.pages:
                            intro += page.extract_text()
                        hero = file_name.split(".")[0]
                        self.hero_data[hero] = intro
                except Exception as e:
                    print(f"读取pdf文件 {file_name} 异常:{e}")
                    continue
        corpus = {}
        self.index_to_name ={}
        index = 0
        for hero,intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index +=1
        self.bm25_model = BM25(corpus)
        return
    def retrieve(self,user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores,key=lambda x:x[1],reverse = True)
        hero = sorted_scores[0][0]
        text = self.hero_data[hero]
        return text
    def query(self,user_query):
        print("user_query", user_query)
        print("-------------------------------")
        retrieve_text = self.retrieve(user_query)
        print("-------------------------------")
        prompt = f"请根据以下从数据库中获得的英雄故事和技能介绍，回答用户问题：\n\n英雄故事及技能介绍：\n{retrieve_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答:", response_text)
        print("-------------------------------")
if __name__ =="__main__":
    rag = SimpleRAG()
    user_query ="黑神话：悟空火爆的基础是什么?"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))