import requests
import json
import time
from typing import Dict, List, Optional

# 智谱API配置（请替换为你的实际信息）
ZHIPU_API_CONFIG = {
    "api_key": "5e6b294f0fa643bb8ceb7994ca440b69.VN55QOPyoriTx88Z",  # 需从智谱AI官网申请
    "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",  # 智谱通用对话API地址
    "model": "glm-3-turbo"  # 可替换为其他智谱模型，如glm-3-turbo等
}


def call_zhipu_model(prompt: str, context: Optional[str] = None) -> str:
    """
    调用智谱大模型API，支持RAG上下文注入
    :param prompt: 用户问题
    :param context: RAG检索到的上下文信息（None表示不注入）
    :return: 模型回答文本
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZHIPU_API_CONFIG['api_key']}"
    }

    # 构建消息列表：注入RAG时添加上下文作为系统提示
    messages = []
    if context:
        messages.append({
            "role": "system",
            "content": f"请基于以下提供的上下文信息回答用户问题，确保信息准确且不添加外部知识：{context}"
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    # 构建请求数据
    data = {
        "model": ZHIPU_API_CONFIG["model"],
        "messages": messages,
        "temperature": 0.1,  # 低随机性，保证对比一致性
        "max_tokens": 500  # 限制最大回答长度
    }

    try:
        response = requests.post(
            url=ZHIPU_API_CONFIG["api_url"],
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        response.raise_for_status()  # 检查HTTP错误
        # 解析智谱API响应（参考官方文档格式）
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"响应格式异常：{result}"
    except Exception as e:
        return f"调用失败：{str(e)}"


# 测试案例设计（覆盖智谱模型可能存在的知识边界）
test_cases: List[Dict] = [
    {
        "case_name": "时效性问题（2024年科技产品）",
        "prompt": "2024年8月发布的华为Mate 70系列有哪些核心升级？",
        "rag_context": "华为Mate 70系列于2024年8月29日发布，核心升级包括：1. 搭载自研麒麟9100芯片，采用4nm+工艺；2. 配备5000mAh电池，支持88W有线快充和50W无线反充；3. 后置四摄系统，主摄为5000万像素超感光镜头，支持8K视频录制；4. 预装HarmonyOS 5.0系统，新增AI办公助手功能。"
    },
    {
        "case_name": "专业领域问题（金融监管政策）",
        "prompt": "2024年中国证监会对量化交易的监管新规有哪些要点？",
        "rag_context": "2024年6月中国证监会发布《量化交易监管指引》，要点包括：1. 要求量化机构注册资本不低于1亿元，实缴资本不低于5000万元；2. 高频交易需报备策略细节，单次申报速度不得低于500微秒；3. 限制量化产品单日换手率不得超过20倍；4. 建立量化交易数据实时报送机制，违者最高罚款1000万元。"
    },
    {
        "case_name": "小众事实问题（地方文化细节）",
        "prompt": "苏州评弹中的「书码头」具体指什么？2024年有哪些新变化？",
        "rag_context": "苏州评弹中的「书码头」原指历史上苏州地区集中演出评弹的茶楼、书场聚集地（以观前街为核心）。2024年3月，苏州文旅局启动「新说书码头」计划，将10处传统书场改造为融合现代声学设计的演出空间，并引入「线上直播+线下体验」模式，全年新增评弹演出场次超500场，吸引年轻观众占比提升至35%。"
    }
]


def run_comparison():
    """执行RAG注入与不注入的对比测试"""
    print(f"===== 智谱大模型（{ZHIPU_API_CONFIG['model']}）RAG效果对比 =====")
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n【测试案例 {i}】：{case['case_name']}")
        print(f"问题：{case['prompt']}")

        # 不注入RAG的回答
        print("\n[不注入RAG] 回答：")
        start_time = time.time()
        no_rag_answer = call_zhipu_model(case["prompt"])
        print(f"响应时间：{time.time() - start_time:.2f}秒")
        print(no_rag_answer)

        # 注入RAG的回答
        print("\n[注入RAG] 回答：")
        start_time = time.time()
        rag_answer = call_zhipu_model(case["prompt"], case["rag_context"])
        print(f"响应时间：{time.time() - start_time:.2f}秒")
        print(rag_answer)

        print("\n" + "-" * 80)  # 分隔线


if __name__ == "__main__":
    run_comparison()
