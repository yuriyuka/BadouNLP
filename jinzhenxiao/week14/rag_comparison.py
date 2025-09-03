#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from zai import ZhipuAiClient

# 智谱AI配置
API_KEY = "daf6fc366ab9xxx" 

# 测试问题
test_questions = [
    "什么是机器学习？",
    "深度学习和机器学习有什么区别？",
    "自然语言处理的主要任务有哪些？", 
    "什么是Transformer架构？",
    "AI在医疗领域有哪些应用？"
]

def call_glm4_without_rag(question: str) -> str:
    """调用GLM-4进行普通问答（非RAG）"""
    client = ZhipuAiClient(api_key=API_KEY)
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=1500,
            stream=False
        )
        
        # 检查响应类型并处理
        print(f"Response type: {type(response)}")
        
        # 如果是tuple，尝试取第一个元素
        if isinstance(response, tuple):
            actual_response = response[0]
            if hasattr(actual_response, 'choices'):
                return actual_response.choices[0].message.content
        elif hasattr(response, 'choices'):
            return response.choices[0].message.content
        
        return str(response)
        
    except Exception as e:
        print(f"非RAG调用错误: {e}")
        return f"调用错误: {e}"

def call_glm4_with_rag(question: str, knowledge_id: str = "your_knowledge_id") -> str:
    """调用GLM-4进行RAG问答（需要先创建知识库）"""
    client = ZhipuAiClient(api_key=API_KEY)
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": question}
            ],
            tools=[
                {
                    "type": "retrieval",
                    "retrieval": {
                        "knowledge_id": knowledge_id,
                        "prompt_template": "从文档\n\"\"\"\n{{knowledge}}\n\"\"\"\n中找问题\n\"\"\"\n{{question}}\n\"\"\"\n的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n不要复述问题，直接开始回答。"
                    }
                }
            ],
            temperature=0.7,
            max_tokens=1500,
            stream=True
        )
        
        # 处理流式响应
        result = ""
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
        
        return result
        
    except Exception as e:
        print(f"RAG调用错误: {e}")
        return f"RAG调用错误: {e}"

def simulate_rag_with_context(question: str) -> str:
    """模拟RAG效果"""
    client = ZhipuAiClient(api_key=API_KEY)

    knowledge_base = """
人工智能基础知识：
人工智能（AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的机器。AI包括机器学习、深度学习、自然语言处理、计算机视觉等多个子领域。机器学习是AI的核心技术之一，通过算法让机器从数据中学习模式。深度学习使用神经网络来处理复杂的模式识别任务。

机器学习算法：
机器学习算法主要分为三类：监督学习、无监督学习和强化学习。监督学习使用标记数据训练模型，如分类和回归问题。无监督学习从无标记数据中发现隐藏模式，如聚类和降维。强化学习通过奖励机制让智能体学习最优策略。常见算法包括线性回归、决策树、随机森林、支持向量机、神经网络等。

自然语言处理：
自然语言处理（NLP）是AI的一个重要分支，专注于让计算机理解和生成人类语言。NLP的主要任务包括文本分类、情感分析、命名实体识别、机器翻译、问答系统等。Transformer架构革命性地改进了NLP性能，BERT、GPT等模型都基于此架构。

深度学习网络：
深度学习使用多层神经网络来学习数据的层次化表示。卷积神经网络（CNN）特别适合处理图像数据。循环神经网络（RNN）和长短期记忆网络（LSTM）适合处理序列数据。注意力机制允许模型关注输入的重要部分，提高了模型性能。

AI应用领域：
人工智能在医疗领域用于疾病诊断、药物发现和个性化治疗。在金融领域，AI用于风险评估、算法交易和反欺诈。在自动驾驶中，AI结合计算机视觉和决策系统实现无人驾驶。在推荐系统中，AI分析用户行为来提供个性化推荐。
"""
    
    # 构建RAG提示词
    rag_prompt = f"""从文档
    \"\"\"
    {knowledge_base}
    \"\"\"
    中找问题
    \"\"\"
    {question}
    \"\"\"
    的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。
    不要复述问题，直接开始回答。"""
    
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            stream=False
        )
        
        # 处理非流式响应
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return str(response)
        
    except Exception as e:
        print(f"模拟RAG调用错误: {e}")
        return f"模拟RAG调用错误: {e}"

def main():
    """主函数：运行RAG效果对比"""
    print("=" * 80)
    print("RAG vs 非RAG 效果对比测试")
    print("=" * 80)
    print("注意：由于没有实际知识库ID，RAG使用模拟方式（本地知识库上下文）")
    print("=" * 80)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 60)
        
        # 非RAG回答
        print("正在获取非RAG回答...")
        non_rag_answer = call_glm4_without_rag(question)
        time.sleep(3)  # 避免请求过快
        
        # RAG回答（模拟）
        print("正在获取RAG回答（模拟）...")
        rag_answer = simulate_rag_with_context(question)
        time.sleep(3)  # 避免请求过快
        
        # 保存结果
        result = {
            "question": question,
            "non_rag_answer": non_rag_answer,
            "rag_answer": rag_answer
        }
        results.append(result)
        
        # 显示回答（截取前500字符）
        print("\n【非RAG回答】:")
        print(non_rag_answer[:500] + "..." if len(non_rag_answer) > 500 else non_rag_answer)
        print("\n【RAG回答（模拟）】:")
        print(rag_answer[:500] + "..." if len(rag_answer) > 500 else rag_answer)
        print("\n" + "="*80)
    
    # 保存结果到文件
    with open("rag_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n对比测试完成！详细结果已保存到 rag_comparison_results.json")
    
    # 生成分析报告
    print("\n" + "="*80)
    print("RAG vs 非RAG 对比分析")
    print("="*80)
    print("主要区别:")
    print("1. 【准确性】RAG基于特定知识库，回答更准确和权威")
    print("2. 【一致性】RAG答案来源可追溯，减少幻觉问题")
    print("3. 【专业性】RAG适合特定领域问答，非RAG更通用")
    print("4. 【时效性】RAG可包含最新信息，非RAG依赖训练数据")
    print("\n建议:")
    print("- 专业领域问答推荐使用RAG")
    print("- 通用知识问答可使用非RAG")
    print("- 结合使用可获得最佳效果")

if __name__ == "__main__":
    main()