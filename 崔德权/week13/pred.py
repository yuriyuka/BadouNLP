# -*- coding: utf-8 -*-

import torch
import json
from transformers import BertTokenizer
from peft import PeftModel
from model import TorchModel
from config import Config

class NERPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.id2label = config["id2label"]
        
        # 加载基础模型
        self.base_model = TorchModel(config)
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def predict(self, text):
        """对输入文本进行NER预测"""
        # 分词
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        
        # 转换为input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # padding
        pad_len = self.config["max_length"] - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
        
        # 转换为tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
        
        # 处理结果
        predictions = predictions[0].cpu().numpy()
        results = []
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            label = self.id2label.get(pred_id, "O")
            results.append({
                "token": token,
                "label": label
            })
        
        return results
    
    def extract_entities(self, text):
        """提取实体"""
        results = self.predict(text)
        entities = []
        current_entity = None
        
        for result in results:
            token = result["token"]
            label = result["label"]
            
            if label.startswith("B-"):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "type": label[2:],  # 去掉B-前缀
                    "start": len(entities)
                }
            elif label.startswith("I-") and current_entity and label[2:] == current_entity["type"]:
                # 继续当前实体
                current_entity["text"] += token
            else:
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities

def main():
    # 测试文本
    test_texts = [
        "张三在北京大学读书",
        "马云创立了阿里巴巴公司",
        "苹果公司在美国加利福尼亚州",
        "诸葛亮是三国时期的军师"
    ]
    
    # 加载模型
    model_path = "output/lora_tuning_ner.pth"
    predictor = NERPredictor(Config, model_path)
    
    print("NER预测结果:")
    print("=" * 50)
    
    for text in test_texts:
        print(f"文本: {text}")
        entities = predictor.extract_entities(text)
        if entities:
            print("实体:")
            for entity in entities:
                print(f"  - {entity['text']} ({entity['type']})")
        else:
            print("  未发现实体")
        print("-" * 30)

if __name__ == "__main__":
    main()
