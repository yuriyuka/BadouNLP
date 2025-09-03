import os
from collections import defaultdict
from typing import Dict, List, Tuple


class BPEStatistics:
    """统计工具类"""
    
    @staticmethod
    def get_token_pair_frequencies(token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        """统计token对的出现频率"""
        frequencies = defaultdict(int)
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            frequencies[pair] += 1
        return frequencies
    
    @staticmethod
    def find_most_frequent_pair(frequencies: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
        """找出频率最高的token对"""
        if not frequencies:
            raise ValueError("频率字典为空")
        return max(frequencies, key=frequencies.get)


class BPEMerger:
    """合并操作工具类"""
    
    @staticmethod
    def merge_token_pair(
        pair: Tuple[int, int], 
        token_ids: List[int], 
        new_token_id: int
    ) -> List[int]:
        """
        将指定的token对合并为新的token
        
        Args:
            pair: 要合并的token对 (token1, token2)
            token_ids: 当前的token ID序列
            new_token_id: 合并后的新token ID
            
        Returns:
            合并后的新token序列
        """
        new_sequence = []
        i = 0
        
        while i < len(token_ids):
            # 检查是否匹配要合并的pair
            if (i < len(token_ids) - 1 and 
                token_ids[i] == pair[0] and 
                token_ids[i + 1] == pair[1]):
                
                new_sequence.append(new_token_id)
                i += 2  # 跳过已合并的token
            else:
                new_sequence.append(token_ids[i])
                i += 1
                
        return new_sequence


class BPETokenizer:
    """BPE分词器主类"""
    
    def __init__(self, vocab_size: int = 256):
        if vocab_size <= 256:
            raise ValueError("词汇表大小必须大于256")
            
        self.vocab_size = vocab_size
        self.merge_rules: Dict[Tuple[int, int], int] = {}  # 合并规则: (token1, token2) -> new_token
        self.reverse_vocab: Dict[int, Tuple[int, int]] = {}  # 反向词汇表: new_token -> (token1, token2)
        self._initialize_base_vocab()
    
    def _initialize_base_vocab(self):
        """初始化基础字节词汇表(0-255)"""
        # 基础词汇表是单字节到自身的映射
        pass
    
    def train(self, training_text: str) -> None:
        """
        训练BPE分词器
        
        Args:
            training_text: 训练文本
        """
        # 将文本编码为字节序列
        token_ids = list(training_text.encode("utf-8"))
        merges_count = self.vocab_size - 256
        
        print(f"开始BPE训练，目标合并次数: {merges_count}")
        
        for merge_index in range(merges_count):
            # 统计当前token对的频率
            frequencies = BPEStatistics.get_token_pair_frequencies(token_ids)
            
            if not frequencies:
                print("没有更多可合并的token对，提前终止训练")
                break
            
            # 找出最频繁的token对
            most_frequent_pair = BPEStatistics.find_most_frequent_pair(frequencies)
            new_token_id = 256 + merge_index
            
            # 执行合并操作
            token_ids = BPEMerger.merge_token_pair(
                most_frequent_pair, token_ids, new_token_id
            )
            
            # 记录合并规则
            self.merge_rules[most_frequent_pair] = new_token_id
            self.reverse_vocab[new_token_id] = most_frequent_pair
            
            print(f"合并 #{merge_index + 1}: {most_frequent_pair} -> {new_token_id}")
        
        print("BPE训练完成")
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token IDs
        
        Args:
            text: 输入文本
            
        Returns:
            token ID列表
        """
        # 初始化为字节序列
        token_ids = list(text.encode("utf-8"))
        
        # 应用所有学到的合并规则
        changed = True
        while changed and len(token_ids) >= 2:
            changed = False
            frequencies = BPEStatistics.get_token_pair_frequencies(token_ids)
            
            if not frequencies:
                break
            
            # 找出当前序列中最频繁且已学习的token对
            applicable_pairs = [
                pair for pair in frequencies.keys() 
                if pair in self.merge_rules
            ]
            
            if not applicable_pairs:
                break
                
            # 选择最频繁的可用pair进行合并
            target_pair = max(
                applicable_pairs, 
                key=lambda p: frequencies[p]
            )
            
            new_token_id = self.merge_rules[target_pair]
            token_ids = BPEMerger.merge_token_pair(
                target_pair, token_ids, new_token_id
            )
            changed = True
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将token IDs解码回文本
        
        Args:
            token_ids: token ID列表
            
        Returns:
            解码后的文本
        """
        current_ids = token_ids[:]  # 创建副本
        
        # 不断展开合并的token，直到所有token都是基础字节
        while True:
            new_ids = []
            has_composite_tokens = False
            
            for token_id in current_ids:
                if token_id >= 256:  # 这是一个合并后的token
                    # 展开为原始的两个token
                    original_pair = self.reverse_vocab[token_id]
                    new_ids.extend(original_pair)
                    has_composite_tokens = True
                else:
                    new_ids.append(token_id)
            
            if not has_composite_tokens:
                break
                
            current_ids = new_ids
        
        # 将字节序列解码为文本
        try:
            return bytes(current_ids).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(current_ids).decode("utf-8", errors="replace")


class CorpusLoader:
    """语料加载工具类"""
    
    @staticmethod
    def load_corpus_from_directory(directory_path: str, file_extension: str = ".txt") -> str:
        """
        从目录加载所有指定扩展名的文本文件
        
        Args:
            directory_path: 目录路径
            file_extension: 文件扩展名
            
        Returns:
            合并后的文本内容
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        corpus_text = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith(file_extension):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        corpus_text.append(file.read())
                except Exception as e:
                    print(f"读取文件 {filename} 时出错: {e}")
        
        if not corpus_text:
            raise ValueError("目录中没有找到有效的文本文件")
        
        return "".join(corpus_text)


def main():
    """主函数示例"""
    # 配置参数
    VOCAB_SIZE = 10000
    CORPUS_DIR = "Heroes"
    
    try:
        # 加载训练语料
        print("正在加载训练语料...")
        training_corpus = CorpusLoader.load_corpus_from_directory(CORPUS_DIR)
        print(f"语料加载完成，总字符数: {len(training_corpus)}")
        
        # 初始化并训练分词器
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.train(training_corpus)
        
        # 测试编码解码
        test_text = "This is the Hugging Face Course..."
        print(f"\n测试文本: {test_text}")
        
        encoded_ids = tokenizer.encode(test_text)
        print(f"编码结果: {encoded_ids}")
        
        decoded_text = tokenizer.decode(encoded_ids)
        print(f"解码结果: {decoded_text}")
        
        # 验证往返一致性
        if test_text == decoded_text:
            print("✓ 编码解码往返测试成功")
        else:
            print("✗ 编码解码往返测试失败")
            
    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()
