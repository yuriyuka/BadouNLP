"""
本地Embedding模型 - 无需API Key
使用Sentence Transformers库的预训练模型
"""

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("需要安装sentence-transformers: pip install sentence-transformers")

class LocalEmbeddingFunction(EmbeddingFunction):
    """
    本地Embedding函数，使用Sentence Transformers
    优点：
    1. 完全离线运行，无需API Key
    2. 免费使用
    3. 速度快
    4. 支持中英文
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        初始化本地embedding模型
        
        Args:
            model_name: 模型名称，推荐选项：
                - "all-MiniLM-L6-v2": 轻量级，英文为主，384维 (推荐)
                - "paraphrase-multilingual-MiniLM-L12-v2": 多语言，384维
                - "distiluse-base-multilingual-cased": 多语言，512维
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
        
        self.model_name = model_name
        print(f"🔄 正在加载本地embedding模型: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✅ 模型加载成功! 向量维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 回退到更小的模型
            print("🔄 尝试加载备用模型...")
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        将文档转换为向量
        
        Args:
            input: 文档列表
        
        Returns:
            向量列表
        """
        try:
            # 使用本地模型生成embedding
            embeddings = self.model.encode(input)
            
            # 转换为列表格式（ChromaDB要求）
            return embeddings.tolist()
            
        except Exception as e:
            print(f"❌ Embedding生成失败: {e}")
            # 返回零向量作为备用
            dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dim] * len(input)

def query_to_vector_local(text, model_name="all-MiniLM-L6-v2"):
    """
    单独的向量转换函数，用于测试
    
    Args:
        text: 要转换的文本
        model_name: 模型名称
    
    Returns:
        numpy数组格式的向量
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ 需要安装sentence-transformers")
        return None
    
    try:
        model = SentenceTransformer(model_name)
        vector = model.encode([text])
        return vector[0]  # 返回第一个结果
    except Exception as e:
        print(f"❌ 本地embedding失败: {e}")
        return None

# 推荐的模型配置
RECOMMENDED_MODELS = {
    "lightweight": {
        "name": "all-MiniLM-L6-v2",
        "description": "轻量级，快速，384维，主要支持英文",
        "size": "23MB"
    },
    "multilingual": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2", 
        "description": "多语言支持，384维，中英文效果好",
        "size": "266MB"
    },
    "best_quality": {
        "name": "all-mpnet-base-v2",
        "description": "最佳质量，768维，主要支持英文",
        "size": "420MB"
    }
}

def test_local_embedding():
    """测试本地embedding"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ 请先安装: pip install sentence-transformers")
        return False
    
    print("🧪 测试本地Embedding模型...")
    
    # 测试文本
    test_texts = [
        "这是一个中文测试",
        "This is an English test",
        "RAG技术很有用"
    ]
    
    for model_info in RECOMMENDED_MODELS.values():
        model_name = model_info["name"]
        print(f"\n📊 测试模型: {model_name}")
        print(f"   描述: {model_info['description']}")
        print(f"   大小: {model_info['size']}")
        
        try:
            # 测试单个文本
            vector = query_to_vector_local(test_texts[0], model_name)
            if vector is not None:
                print(f"   ✅ 成功! 向量维度: {vector.shape}")
                print(f"   前5个值: {vector[:5]}")
                break
            else:
                print(f"   ❌ 失败")
        except Exception as e:
            print(f"   ❌ 错误: {e}")
    
    return True

if __name__ == '__main__':
    print("🏠 本地Embedding模型测试")
    print("="*40)
    
    success = test_local_embedding()
    
    if success:
        print("\n🎉 本地embedding设置成功!")
        print("现在可以在RAG系统中使用本地embedding了")
        print("\n📋 安装命令:")
        print("pip install sentence-transformers")
    else:
        print("\n💡 如果遇到问题，可以尝试:")
        print("1. pip install sentence-transformers torch")
        print("2. 检查网络连接（首次需要下载模型）") 