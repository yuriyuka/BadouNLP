import requests
import json
import time
import os
from typing import Dict, List, Optional

# 智谱API配置 - 从环境变量读取更安全
ZHIPU_API_CONFIG = {
    "api_key": os.environ.get("ZHIPU_API_KEY", "b2c68007279b4a30904a46c2e29afa81.W68kUXlXwBgDLXfM"),  # 从环境变量获取
    "api_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "model": "glm-3-turbo"
}


def call_zhipu_model(prompt: str, context: Optional[str] = None) -> str:
    """
    调用智谱大模型API，支持RAG上下文注入
    :param prompt: 用户问题
    :param context: RAG检索到的上下文信息（None表示不注入）
    :return: 模型回答文本
    """
    # 检查API密钥
    if ZHIPU_API_CONFIG["api_key"] == "b2c68007279b4a30904a46c2e29afa81.W68kUXlXwBgDLXfM":
        return "错误：请先设置ZHIPU_API_KEY环境变量或修改代码中的API密钥"
    
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
        
        # 解析智谱API响应
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"API错误：{result['error'].get('message', '未知错误')}"
        else:
            return f"响应格式异常：{result}"
    except requests.exceptions.Timeout:
        return "请求超时，请检查网络连接"
    except requests.exceptions.RequestException as e:
        return f"网络请求失败：{str(e)}"
    except Exception as e:
        return f"处理响应时出错：{str(e)}"


# 更新测试案例（涵盖科技、医疗健康和历史文化领域）
test_cases: List[Dict] = [
    {
        "case_name": "前沿科技（量子计算）",
        "prompt": "2024年量子计算领域有哪些突破性进展？",
        "rag_context": "2024年量子计算领域取得多项突破：1. 谷歌研发的'悬铃木'量子处理器实现512量子比特，错误率降低至0.001%；2. 中国科研团队成功开发出首台量子计算机专用操作系统'量羲OS'；3. IBM与多家制药公司合作，利用量子计算加速新药研发，将分子模拟时间从数周缩短至几小时；4. 量子加密通信实现1000公里距离的安全传输，创下新纪录。"
    },
    {
        "case_name": "医疗健康（基因编辑）",
        "prompt": "CRISPR基因编辑技术最近有哪些临床应用？",
        "rag_context": "2024年CRISPR基因编辑技术临床应用进展：1. FDA批准首款基于CRISPR的遗传病疗法'Casgevy'用于治疗镰状细胞病和β地中海贫血；2. 中国研究人员成功使用CRISPR技术治疗先天性黑蒙症，首批5名患者视力显著改善；3. 新型'碱基编辑'技术实现更精准的基因修复，副作用降低80%；4. 研究人员开发出'迷你CRISPR'系统，体积减小50%，更适合体内递送。"
    },
    {
        "case_name": "历史文化（考古发现）",
        "prompt": "最近三星堆遗址有哪些重要考古发现？",
        "rag_context": "2024年三星堆遗址考古取得重大进展：1. 新发现的8号祭祀坑出土黄金面具残片重达280克，是已发现最大金面具；2. 发现丝绸残留物证据，将四川地区丝绸使用历史提前至距今3000多年前；3. 出土的龟背形网格器经CT扫描发现内有玉器，是首次发现的青铜网格器；4. 发现大量象牙雕刻残片，证实古蜀国与东南亚存在贸易往来；5. 碳14测年显示新发现文物年代为商代晚期，约公元前1200-1000年。"
    },
    {
        "case_name": "商业经济（人工智能监管）",
        "prompt": "欧盟人工智能法案对生成式AI有哪些具体规定？",
        "rag_context": "欧盟人工智能法案于2024年5月正式实施，对生成式AI的规定包括：1. 要求所有生成式AI系统必须明确标注内容为AI生成；2. 禁止使用未经授权的受版权保护内容训练AI模型；3. 高风险AI系统需进行强制性基本权利影响评估；4. 建立生成式AI数据库，记录所有训练数据来源；5. 违规企业最高可处全球年营业额6%的罚款。"
    }
]


def run_comparison():
    """执行RAG注入与不注入的对比测试"""
    print(f"===== 智谱大模型（{ZHIPU_API_CONFIG['model']}）RAG效果对比 =====")
    print(f"测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n【测试案例 {i}】：{case['case_name']}")
        print(f"问题：{case['prompt']}")

        # 不注入RAG的回答
        print("\n[不注入RAG] 回答：")
        start_time = time.time()
        no_rag_answer = call_zhipu_model(case["prompt"])
        response_time = time.time() - start_time
        print(f"响应时间：{response_time:.2f}秒")
        print(no_rag_answer)

        # 注入RAG的回答
        print("\n[注入RAG] 回答：")
        start_time = time.time()
        rag_answer = call_zhipu_model(case["prompt"], case["rag_context"])
        response_time = time.time() - start_time
        print(f"响应时间：{response_time:.2f}秒")
        print(rag_answer)

        print("\n" + "-" * 80)  # 分隔线


if __name__ == "__main__":
    run_comparison()
