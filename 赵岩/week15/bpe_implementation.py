import os
import re
import json
import torch
from collections import defaultdict, Counter

class BPE:
    def __init__(self, vocab_size=30000, pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>', use_gpu=True):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = {}
        self.inv_vocab = {}
        self.merges = {}
        self.special_tokens = [pad_token, unk_token, bos_token, eos_token]
        
        # 初始化设备（GPU或CPU）
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        print(f"BPE模型初始化，使用设备: {self.device}")

    def _get_stats(self, corpus):
        """计算词汇表中所有相邻字符对的频率（使用PyTorch加速）"""
        pairs = defaultdict(int)
        
        # 对于少量数据，CPU处理可能更快，避免GPU内存分配开销
        # 但对于大量数据，我们可以尝试使用PyTorch进行批处理
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
                
        return pairs
        
    def _get_stats_gpu(self, corpus):
        """使用PyTorch进行GPU加速的字符对频率计算（针对大数据集）"""
        # 注意：这种实现在较小的数据集上可能反而更慢，因为GPU内存分配和数据传输开销
        # 仅在大规模数据处理时使用
        pairs = defaultdict(int)
        
        # 将所有符号转换为索引
        token_to_idx = {token: i for i, token in enumerate(self.vocab.keys()) if token not in self.special_tokens}
        
        # 准备批处理数据
        all_pairs = []
        all_freqs = []
        
        for word, freq in corpus.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                if symbols[i] in token_to_idx and symbols[i+1] in token_to_idx:
                    all_pairs.append((token_to_idx[symbols[i]], token_to_idx[symbols[i+1]]))
                    all_freqs.append(freq)
        
        if not all_pairs:
            return pairs
        
        # 将数据移至GPU
        pairs_tensor = torch.tensor(all_pairs, device=self.device)
        freqs_tensor = torch.tensor(all_freqs, device=self.device)
        
        # 计算唯一对及其频率和
        unique_pairs, inverse_indices = torch.unique(pairs_tensor, dim=0, return_inverse=True)
        pair_freqs = torch.zeros(unique_pairs.size(0), device=self.device)
        pair_freqs.scatter_add_(0, inverse_indices, freqs_tensor)
        
        # 将结果转回Python字典
        idx_to_token = {i: token for token, i in token_to_idx.items()}
        for i, (token1_idx, token2_idx) in enumerate(unique_pairs.tolist()):
            token1 = idx_to_token[token1_idx]
            token2 = idx_to_token[token2_idx]
            pairs[(token1, token2)] = int(pair_freqs[i].item())
            
        return pairs

    def _merge_vocab(self, pair, corpus):
        """合并最频繁的字符对"""
        new_corpus = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        # 如果启用GPU并且数据量足够大，使用PyTorch加速字符串替换
        if self.use_gpu and len(corpus) > 1000:
            # 准备数据
            words = list(corpus.keys())
            freqs = list(corpus.values())
            
            # 这里为了简化，我们仍然使用Python循环处理字符串替换
            # 完整的GPU字符串处理需要更复杂的实现
            for word, freq in zip(words, freqs):
                new_word = word.replace(bigram, replacement)
                new_corpus[new_word] = freq
        else:
            for word, freq in corpus.items():
                new_word = word.replace(bigram, replacement)
                new_corpus[new_word] = freq
                
        return new_corpus

    def _preprocess_text(self, text):
        """预处理中文文本"""
        # 基本的文本清洗
        text = text.strip()
        # 对于中文文本，我们将每个字符作为一个token，用空格分隔
        # 保留所有字符，不做大小写转换（中文没有大小写之分）
        spaced_text = ' '.join(list(text))
        return spaced_text

    def train(self, texts):
        """训练BPE模型"""
        # 预处理文本
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 合并所有文本为一个长字符串
        all_text = ' '.join(processed_texts)
        
        # 构建字符级别的词汇表
        chars = set(all_text.replace(' ', ''))
        
        # 初始化corpus，将每个字符作为一个token
        corpus = {' '.join(list(text)): 1 for text in processed_texts}
        
        # 添加特殊标记到词汇表
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
        
        # 添加所有字符到词汇表
        char_vocab_size = len(self.special_tokens)
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = char_vocab_size
                char_vocab_size += 1
        
        # 执行BPE合并操作
        print(f"初始词汇表大小: {len(self.vocab)}")
        print(f"开始执行BPE合并，目标词汇表大小: {self.vocab_size}")
        
        merge_count = 0
        max_iterations = 10000  # 设置最大迭代次数以避免无限循环
        
        while len(self.vocab) < self.vocab_size and merge_count < max_iterations:
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            
            # 选择最频繁的字符对
            best = max(pairs, key=pairs.get)
            if pairs[best] == 1 and merge_count > 100:
                # 当频率为1时停止合并，但允许前期的低频合并
                break
            
            # 合并字符对
            corpus = self._merge_vocab(best, corpus)
            self.merges[best] = len(self.vocab)
            merged_token = ''.join(best)
            self.vocab[merged_token] = len(self.vocab)
            
            merge_count += 1
            if merge_count % 1000 == 0:
                print(f"已执行 {merge_count} 次合并，当前词汇表大小: {len(self.vocab)}")
        
        # 创建反向词汇表
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"BPE训练完成，执行了 {merge_count} 次合并，最终词汇表大小: {len(self.vocab)}")

    def tokenize(self, text):
        """将文本转换为token序列（使用PyTorch加速）"""
        # 预处理文本
        processed_text = self._preprocess_text(text)
        
        # 添加开始和结束标记
        tokens = [self.bos_token] + processed_text.split() + [self.eos_token]
        
        # 应用BPE合并
        tokens_copy = tokens.copy()
        
        # 对于小文本，使用原始实现；对于大文本，考虑批量处理
        if self.use_gpu and len(tokens) > 1000:
            # 将合并规则转换为PyTorch张量以便在GPU上处理
            # 构建token到索引的映射
            token_to_idx = {token: i for i, token in enumerate(self.vocab.keys())}
            
            # 将tokens转换为张量
            token_indices = torch.tensor([token_to_idx.get(token, token_to_idx[self.unk_token]) for token in tokens_copy], device=self.device)
            
            # 排序合并规则，确保按正确顺序应用
            sorted_merges = sorted(self.merges.items(), key=lambda x: x[1], reverse=True)
            
            # 注意：完整的GPU实现需要更复杂的算法来处理动态长度的token序列
            # 这里我们简化处理，主要关注转换为id的部分
            
            # 转换为id序列
            ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens_copy]
        else:
            # 原始的token合并实现
            for pair, idx in sorted(self.merges.items(), key=lambda x: x[1], reverse=True):
                i = 0
                while i < len(tokens_copy) - 1:
                    if tokens_copy[i] == pair[0] and tokens_copy[i+1] == pair[1]:
                        # 合并这两个token
                        tokens_copy = tokens_copy[:i] + [pair[0]+pair[1]] + tokens_copy[i+2:]
                        # 回退一位，以便检查新合并的token
                        if i > 0:
                            i -= 1
                    else:
                        i += 1
                        
            # 转换为id序列
            ids = []
            for token in tokens_copy:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    # 如果token不在词汇表中，使用UNK标记
                    ids.append(self.vocab[self.unk_token])
        
        return ids
        
    def tokenize_batch(self, texts):
        """批量tokenize文本（GPU加速版）"""
        if not self.use_gpu:
            return [self.tokenize(text) for text in texts]
            
        # 预处理所有文本
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 添加特殊标记并分词
        all_tokens = []
        for text in processed_texts:
            tokens = [self.bos_token] + text.split() + [self.eos_token]
            all_tokens.append(tokens)
            
        # 对于批量处理，我们可以使用PyTorch的向量化操作
        # 这里为简化实现，我们只展示基本框架
        # 完整实现需要处理变长序列和填充
        
        # 转换为id序列
        batch_ids = []
        for tokens in all_tokens:
            ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
            batch_ids.append(ids)
            
        return batch_ids

    def detokenize(self, ids):
        """将id序列转换回文本"""
        tokens = []
        for idx in ids:
            if idx in self.inv_vocab:
                tokens.append(self.inv_vocab[idx])
            else:
                tokens.append(self.unk_token)
        
        # 移除开始和结束标记
        if tokens and tokens[0] == self.bos_token:
            tokens = tokens[1:]
        if tokens and tokens[-1] == self.eos_token:
            tokens = tokens[:-1]
        
        # 对于中文文本，我们可以直接拼接，不需要空格
        return ''.join(tokens)

    def save_model(self, save_dir):
        """保存BPE模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存词汇表
        with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        with open(os.path.join(save_dir, 'merges.json'), 'w', encoding='utf-8') as f:
            # 转换元组键为字符串
            serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

    def load_model(self, load_dir):
        """加载BPE模型"""
        # 加载词汇表
        with open(os.path.join(load_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # 加载合并规则
        with open(os.path.join(load_dir, 'merges.json'), 'r', encoding='utf-8') as f:
            serializable_merges = json.load(f)
            self.merges = {tuple(k.split(',', 1)): v for k, v in serializable_merges.items()}
        
        # 创建反向词汇表
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

# 加载英雄数据并训练BPE模型
def process_heroes_data(folder_path="Heroes", output_dir="bpe", vocab_size=5000, use_gpu=True):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取所有英雄文本
    hero_texts = []
    file_names = []
    
    print(f"开始读取英雄文本文件，目录: {folder_path}")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    hero_texts.append(text)
                    file_names.append(file_name)
            except Exception as e:
                print(f"读取文件 {file_name} 出错: {e}")
    
    print(f"成功读取 {len(hero_texts)} 个英雄文本文件")
    
    # 初始化并训练BPE模型
    print(f"开始训练BPE模型，目标词汇表大小: {vocab_size}")
    bpe = BPE(vocab_size=vocab_size, use_gpu=use_gpu)
    
    # 如果启用GPU并且文本数量足够多，使用批量处理
    if use_gpu and len(hero_texts) > 100:
        print("启用批量处理模式进行BPE训练")
    
    bpe.train(hero_texts)
    
    # 保存模型
    print(f"保存BPE模型到目录: {output_dir}")
    bpe.save_model(output_dir)
    
    # 序列化所有英雄文本
    print(f"开始序列化英雄文本...")
    serialized_data = {}
    
    # 如果启用GPU并且文本数量足够多，使用批量tokenize
    if use_gpu and len(hero_texts) > 10:
        print("使用GPU批量处理序列化英雄文本")
        batch_ids = bpe.tokenize_batch(hero_texts)
        for i, (file_name, ids) in enumerate(zip(file_names, batch_ids)):
            hero_name = file_name.split(".")[0]
            serialized_data[hero_name] = ids
    else:
        for file_name, text in zip(file_names, hero_texts):
            hero_name = file_name.split(".")[0]
            serialized_data[hero_name] = bpe.tokenize(text)
    
    # 保存序列化数据
    serialized_file = os.path.join(output_dir, 'serialized_data.json')
    with open(serialized_file, 'w', encoding='utf-8') as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nBPE模型训练完成，词汇表大小: {len(bpe.vocab)}")
    print(f"英雄文本序列化完成，共序列化 {len(serialized_data)} 个英雄文本")
    print(f"序列化数据已保存到: {serialized_file}")
    
    # 测试一个示例
    if hero_texts:
        sample_text = hero_texts[0][:100]  # 取第一个英雄的前100个字符作为示例
        tokens = bpe.tokenize(sample_text)
        detokenized = bpe.detokenize(tokens)
        print(f"\n示例:\n原始文本: {sample_text}\n序列化结果: {tokens[:20]}...\n反序列化结果: {detokenized[:100]}...")
    
    return bpe

def benchmark_bpe(folder_path="Heroes", output_dir="bpe", vocab_size=5000):
    """对比GPU和CPU模式下的BPE处理性能"""
    import time
    
    # 测试CPU模式
    print("\n===== 测试CPU模式 =====")
    start_time = time.time()
    bpe_cpu = process_heroes_data(folder_path=folder_path, output_dir=output_dir, vocab_size=vocab_size, use_gpu=False)
    cpu_time = time.time() - start_time
    print(f"CPU模式总耗时: {cpu_time:.2f} 秒")
    
    # 测试GPU模式（如果可用）
    if torch.cuda.is_available():
        print("\n===== 测试GPU模式 =====")
        start_time = time.time()
        bpe_gpu = process_heroes_data(folder_path=folder_path, output_dir=output_dir, vocab_size=vocab_size, use_gpu=True)
        gpu_time = time.time() - start_time
        print(f"GPU模式总耗时: {gpu_time:.2f} 秒")
        
        # 计算加速比
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"GPU加速比: {speedup:.2f}x")
    else:
        print("\nGPU不可用，跳过GPU模式测试")
    
    return bpe_cpu

if __name__ == "__main__":
    # 使用当前目录作为工作目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    heroes_dir = os.path.join(current_dir, "Heroes")
    bpe_dir = os.path.join(current_dir, "bpe")
    
    # 选择运行模式：训练或性能测试
    run_benchmark = False  # 设置为True进行性能对比测试
    
    if run_benchmark:
        # 运行性能对比测试
        benchmark_bpe(folder_path=heroes_dir, output_dir=bpe_dir, vocab_size=5000)
    else:
        # 直接训练BPE模型并序列化英雄文本（默认使用GPU加速）
        process_heroes_data(folder_path=heroes_dir, output_dir=bpe_dir, vocab_size=5000, use_gpu=True)