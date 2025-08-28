#!/usr/bin/env python3
"""
DeepSeek R1 Model Query Script

This script allows you to interact with DeepSeek R1 models in GGUF format only.
Supported: DeepSeek-R1-Distill-Qwen-32B (GGUF)
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
import json
import subprocess

def check_dependencies(model_type: str):
    """Check if required dependencies for the selected model are installed."""
    missing_deps = []
    
    if model_type == "gguf":
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            missing_deps.append("llama-cpp-python")
    
    if missing_deps:
        print("Missing dependencies for model type '" + model_type + "'. Please install:")
        for dep in missing_deps:
            print(f"  pip install {dep}")
        if model_type == "gguf" and sys.platform == "darwin":
            print("  # On Apple Silicon for GPU acceleration:")
            print("  CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
        return False
    return True

class GGUFModel:
    """Wrapper for GGUF model using llama-cpp-python."""
    
    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=os.cpu_count(),
                n_gpu_layers=-1,  # Use all available GPU layers
                verbose=False
            )
            print(f"✅ Loaded GGUF model: {os.path.basename(model_path)}")
        except Exception as e:
            raise Exception(f"Failed to load GGUF model: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response using GGUF model."""
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|endoftext|>", "<|im_end|>"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error generating response: {e}"

def find_models():
    """Find available models in the workspace."""
    models = {}
    
    # Check for GGUF model
    gguf_path = "/Users/evan/DeepSeek-R1-Distill-Qwen-32B/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_S.gguf"
    if os.path.exists(gguf_path):
        models["gguf"] = {
            "name": "DeepSeek-R1-Distill-Qwen-32B (GGUF)",
            "path": gguf_path,
            "type": "gguf"
        }
    
    # MLX support removed
    
    return models

def call_rag_system(query: str) -> Dict[str, Any]:
    """调用外部RAG系统获取相关文档"""
    rag_script_path = "/Users/evan/Downloads/AINLP/week14 大语言模型相关第四讲/RAG/dota2英雄介绍-byRAG/vec_db_rag_local.py"
    
    if not os.path.exists(rag_script_path):
        print(f"⚠️  RAG脚本不存在: {rag_script_path}")
        return {"context": "", "retrieved_docs": [], "error": "RAG script not found"}
    
    try:
        # 创建一个临时Python脚本来调用RAG系统
        temp_script = f"""
import sys
sys.path.append('/Users/evan/Downloads/AINLP/week14 大语言模型相关第四讲/RAG/dota2英雄介绍-byRAG')

# 导入RAG模块
import chromadb
from chromadb.config import Settings
from local_embedding import LocalEmbeddingFunction

# 初始化ChromaDB
client = chromadb.Client(Settings(
    persist_directory="./chroma_db_local",
    is_persistent=True
))

# 获取集合
collection = client.get_collection("rag_local")

# 检索文档
results = collection.query(
    query_texts=["{query}"],
    n_results=3
)

# 返回结果
retrieved_docs = results["documents"][0] if results["documents"] else []
context = "\\n".join(retrieved_docs) if retrieved_docs else ""

print("RAG_RESULT_START")
print(context)
print("RAG_RESULT_END")
"""
        
        # 执行临时脚本
        result = subprocess.run(
            [sys.executable, "-c", temp_script],
            capture_output=True,
            text=True,
            cwd="/Users/evan/Downloads/AINLP/week14 大语言模型相关第四讲/RAG/dota2英雄介绍-byRAG"
        )
        
        if result.returncode == 0:
            # 解析输出
            output = result.stdout
            if "RAG_RESULT_START" in output and "RAG_RESULT_END" in output:
                start_idx = output.find("RAG_RESULT_START") + len("RAG_RESULT_START")
                end_idx = output.find("RAG_RESULT_END")
                context = output[start_idx:end_idx].strip()
                
                return {
                    "context": context,
                    "retrieved_docs": context.split("\n") if context else [],
                    "success": True
                }
        
        print(f"⚠️  RAG调用失败: {result.stderr}")
        return {"context": "", "retrieved_docs": [], "error": result.stderr}
        
    except Exception as e:
        print(f"⚠️  RAG调用异常: {e}")
        return {"context": "", "retrieved_docs": [], "error": str(e)}

def create_rag_prompt(query: str, rag_result: Dict[str, Any]) -> str:
    """创建包含RAG上下文的提示"""
    context = rag_result.get("context", "")
    
    if not context:
        return query
    
    # 构建RAG增强的提示
    rag_prompt = f"""基于以下检索到的相关信息来回答问题：

检索到的相关信息：
{context}

用户问题：{query}

请基于上述检索到的信息来回答用户的问题。如果检索到的信息不足以回答问题，请说明并尽可能提供有用的回答。"""
    
    return rag_prompt

def interactive_mode(model, use_rag=False):
    """Run interactive chat mode."""
    print("\n🤖 Interactive mode started. Type 'quit' or 'exit' to end the session.")
    if use_rag:
        print("🔍 RAG mode enabled - queries will be enhanced with retrieved context")
    print("Type your questions or prompts below:\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # 如果启用RAG，先调用RAG系统
            if use_rag:
                print("🔍 正在检索相关文档...")
                rag_result = call_rag_system(user_input)
                
                if rag_result.get("success"):
                    print(f"📄 检索到 {len(rag_result.get('retrieved_docs', []))} 个相关文档")
                    enhanced_prompt = create_rag_prompt(user_input, rag_result)
                else:
                    print("⚠️  RAG检索失败，使用原始查询")
                    enhanced_prompt = user_input
            else:
                enhanced_prompt = user_input
            
            print("🤖 DeepSeek: ", end="", flush=True)
            response = model.generate(enhanced_prompt)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

sys.path.append('/Users/evan/Downloads/AINLP/week14 大语言模型相关第四讲/RAG/dota2英雄介绍-byRAG')

from vec_db_rag_local import get_rag_prompt

def build_prompt_for_model(user_query: str) -> str:
    # 这里会触发 RAG 检索 + 汇总，并返回可直接喂给模型的提示词
    return get_rag_prompt(user_query, top_k=3)

def main():
    parser = argparse.ArgumentParser(description="Query DeepSeek R1 models")
    parser.add_argument("--model", choices=["gguf"], help="Model type to use")
    parser.add_argument("--prompt", "-p", help="Single prompt to process")
    parser.add_argument("--max-tokens", "-m", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--rag", action="store_true", help="Enable RAG mode to enhance queries with retrieved context")
    
    args = parser.parse_args()
    
    # Find available models
    models = find_models()
    
    if not models:
        print("❌ No models found in the workspace!")
        print("Make sure you have the model files in the expected locations:")
        print("  - GGUF: bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/")
        sys.exit(1)
    
    # Select model
    model_type = args.model
    if not model_type:
        if len(models) == 1:
            model_type = list(models.keys())[0]
        else:
            print("Available models:")
            for key, model_info in models.items():
                print(f"  {key}: {model_info['name']}")
            model_type = input("Select model type (gguf): ").strip().lower()
    
    if model_type not in models:
        print(f"❌ Invalid model type: {model_type}")
        sys.exit(1)
    
    model_info = models[model_type]
    print(f"📦 Loading {model_info['name']}...")
    
    # Load model
    try:
        if not check_dependencies(model_type):
            sys.exit(1)
        model = GGUFModel(model_info["path"])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Process input
    if args.interactive:
        interactive_mode(model, use_rag=args.rag)
    elif args.prompt:
        if args.rag:
            print("🔍 正在检索相关文档...")
            rag_result = call_rag_system(args.prompt)
            if rag_result.get("success"):
                print(f"📄 检索到 {len(rag_result.get('retrieved_docs', []))} 个相关文档")
                enhanced_prompt = create_rag_prompt(args.prompt, rag_result)
            else:
                print("⚠️  RAG检索失败，使用原始查询")
                enhanced_prompt = args.prompt
        else:
            enhanced_prompt = args.prompt
        
        print(f"🤖 Response: {model.generate(enhanced_prompt, args.max_tokens, args.temperature)}")
    else:
        # Default to interactive mode
        interactive_mode(model, use_rag=args.rag)

if __name__ == "__main__":
    main()
