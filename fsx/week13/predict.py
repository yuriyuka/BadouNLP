import json
import torch
import torch.nn
from transformers import AutoModelForTokenClassification, BertTokenizer

from config import Config

class PredictorBertNer:
    def __init__(self, config, model_path):
        self.config = config
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        # 复用训练时的加载逻辑
        self.schema = load_schema(config["schema_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.schema_type = {v: k for k, v in self.schema.items()}

    def encode_sentence(self, text):
        """字符粒度编码，与训练时完全一致"""
        input_id = []
        # 按字符遍历（因vocab_path不是words.txt，无需jieba）
        for char in text:
            if self.config["model_type"] == "bert":
                # 复用BERT的vocab映射逻辑（与训练一致）
                token_id = self.tokenizer.vocab.get(char, self.tokenizer.vocab.get("[UNK]"))
            else:
                # 非BERT模型时用自定义vocab
                token_id = self.vocab.get(char, self.vocab["[UNK]"])
            input_id.append(token_id)

        # 严格padding到max_length（与训练时一致）
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id, pad_token=0):
        """与训练时相同的padding逻辑"""
        if len(input_id) < self.config["max_length"]:
            input_id.extend([pad_token] * (self.config["max_length"] - len(input_id)))
        else:
            input_id = input_id[:self.config["max_length"]]
        return input_id

    def predict(self, sentence):
        input_ids = self.encode_sentence(sentence)
        input_ids = torch.tensor([input_ids])

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids)

            # 修正：处理元组形式的输出，提取logits
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 通常第一个元素是logits
            else:
                logits = outputs.logits  # 如果是对象形式则直接取logits

        pred_result = []
        for i in range(len(sentence)):
            pred_label_id = logits[0, i].argmax().item()
            pred_tag = self.schema_type.get(pred_label_id, "O")
            pred_result.append(f"{sentence[i]}:{pred_tag}")

        return " ".join(pred_result)


def load_schema(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict


if __name__ == '__main__':
    for i in range(10):
        predictor = PredictorBertNer(Config, "model_output/lora_ner_model")
        test_sentence = "中共中央党委政治局写给约翰内斯的电报"
        print(predictor.predict(test_sentence))
        print(i)
