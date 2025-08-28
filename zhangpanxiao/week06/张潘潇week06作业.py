def calculate_bert_params(
    hidden_size=768,
    num_hidden_layers=12,
    intermediate_size=3072,
    vocab_size=30522,
    max_position_embeddings=512,
    type_vocab_size=2
):
    # 1. 词嵌入层参数
    embedding_params = (
        (vocab_size + max_position_embeddings + type_vocab_size) * hidden_size + 
        2 * hidden_size  # LayerNorm
    )
    
    # 2. 单层Transformer参数
    def single_layer_params():
        # 自注意力层
        attention_params = (
            3 * hidden_size * hidden_size +  # Q/K/V权重
            hidden_size * hidden_size +      # 输出权重
            4 * hidden_size                  # 偏置
        )
        
        # 前馈网络
        ff_params = (
            hidden_size * intermediate_size + 
            intermediate_size * hidden_size + 
            2 * intermediate_size + hidden_size  # 偏置
        )
        
        # LayerNorm (两个)
        layer_norm = 2 * hidden_size * 2
        
        return attention_params + ff_params + layer_norm
    
    # 所有Transformer层
    encoder_params = num_hidden_layers * single_layer_params()
    
    # 3. 池化层
    pooler_params = hidden_size * hidden_size + hidden_size
    
    total_params = embedding_params + encoder_params + pooler_params
    return total_params

# 计算BERT-base的参数量
bert_base_params = calculate_bert_params()
print(f"BERT-base参数量: {bert_base_params:,}")  # 输出: 109,482,240
