import json

import torch
import torch.nn
from peft import get_peft_model, LoraConfig
from transformers import BertTokenizer

from config import Config
from model import TorchModel


class PredictorBertNer:
    def __init__(self, config, model_path):
        self.config = config
        self.base_model = TorchModel
        self.peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
        self.model = get_peft_model(self.base_model, self.peft_config)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # self.vocab = self.load_vocab(config["bert_vocab_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.schema_type = dict((y, x) for x, y in self.schema.items())

    def load_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                vocab[line.strip()] = index
        return vocab

    def load_schema(self, path):
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return schema

    def predict(self, sentence):
        in_put = []
        for word in sentence:
            in_put.append(self.tokenizer.vocab.get(word, self.tokenizer.vocab["[UNK]"]))
        print(in_put)
        with torch.no_grad():
            in_put = torch.tensor([in_put])
            pred = self.model(in_put)
            print(pred)

        load_sentence = ""
        squeeze = pred[0].squeeze()
        tags = []
        for i in range(len(squeeze)):
            tag = self.schema_type[int(squeeze[i].argmax())]
            tags.append(int(squeeze[i].argmax()))
            load_sentence += sentence[i] + ":" + tag + " "
        print(tags)
        return load_sentence


if __name__ == '__main__':
    for i in range(10):
        predictor = PredictorBertNer(Config, "model_output/epoch_10.pth")

        str = "中共中央党委政治局写给约翰内斯的电报"
        print(predictor.predict(str))

