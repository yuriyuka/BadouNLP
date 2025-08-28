import json
import os
import jieba
import numpy as np
import gradio as gr
from zai import ZhipuAiClient
from bm25 import BM25


# 智谱API调用函数
def call_large_model(prompt):
    client = ZhipuAiClient(api_key=os.environ.get("zhipuApiKey"))
    response = client.chat.completions.create(
        model="glm-4.5",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text


class SimpleRAG:
    def __init__(self, folder_path="Heroes"):
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
        return text, hero

    def query(self, user_query):
        retrive_text, hero_name = self.retrive(user_query)
        prompt = f"请根据以下从数据库中获得的英雄故事和技能介绍，回答用户问题：\n\n英雄故事及技能介绍：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        return response_text, hero_name, retrive_text


# 创建RAG实例
rag = SimpleRAG()


# 对比函数
def compare_rag_vs_direct(query):
    # 使用RAG的回答
    rag_response, hero_name, retrieved_text = rag.query(query)

    # 直接请求大模型的回答
    direct_response = call_large_model(query)

    # 格式化输出
    result = f"""
# RAG vs 直接大模型查询对比

## 用户问题:
{query}

---

## RAG检索到的英雄: {hero_name}

### 检索到的内容:
{retrieved_text[:500]}... [内容已截断]

---

## RAG增强的回答:
{rag_response}

---

## 直接大模型回答:
{direct_response}

---

## 对比分析:
- **RAG增强回答**基于检索到的具体英雄信息，提供更准确、详细的回答
- **直接大模型回答**依赖于模型的内置知识，可能不够精确或过时
- 对于游戏角色、技能等具体信息，RAG能提供更可靠的答案
"""
    return result


# 创建Gradio界面
demo = gr.Interface(
    fn=compare_rag_vs_direct,
    inputs=gr.Textbox(
        lines=2,
        label="输入关于Dota2英雄的问题",
        value="高射火炮是谁的技能？这个技能有什么特点？"
    ),
    outputs=gr.Markdown(
        label="RAG vs 直接大模型回答对比"
    ),
    title="Dota2英雄信息查询 - RAG vs 直接大模型对比",
    description="输入关于Dota2英雄、技能或背景故事的问题，查看RAG增强回答与直接大模型回答的差异",
    examples=[
        ["高射火炮是谁的技能？"],
        ["斧王的背景故事是什么？"],
        ["哪个英雄有闪烁技能？"],
        ["幻影刺客的技能有哪些？"]
    ]
)

if __name__ == "__main__":
    demo.launch()
