from transformers import BertTokenizer

# 初始化BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 示例文本
texts = [
    "Hello, how are you?",
    "I am fine, thank you! How about yourself?"
]

# 构建词表并序列化文本
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 打印结果
print("Input IDs:", encoded_texts['input_ids'])
print("Attention Masks:", encoded_texts['attention_mask'])


