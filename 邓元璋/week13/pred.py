import torch
import logging
import os
from model import TorchModel
from peft import get_peft_model, LoraConfig
from evaluate import Evaluator, load_label_map
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = Config
label_to_index, index_to_label = load_label_map()
id2label = {v: k for k, v in label_to_index.items()}

# 加载模型
model = TorchModel
if config["tuning_tactics"] == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        task_type="TOKEN_CLASSIFICATION"
    )
    model = get_peft_model(model, peft_config)

# 加载权重
model.load_state_dict(torch.load(os.path.join(config["model_path"], "lora_ner.pth")))
model.eval()
model = model.cpu()

# 测试评估
evaluator = Evaluator(config, model, logger)
evaluator.eval(0)


# 单句预测示例
def predict(text):
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])
    inputs = tokenizer(
        list(text),  # 按字符拆分
        is_split_into_words=True,
        max_length=config["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        # 关键修改：从元组中提取logits（通常是第0个元素）
        pred_logits = outputs[0][0]  # 取第一个样本的logits

    pred_labels = torch.argmax(pred_logits, dim=1).numpy()
    # 映射标签并过滤padding
    word_ids = inputs.word_ids(batch_index=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    result = []
    for token, label_id, word_id in zip(tokens, pred_labels, word_ids):
        if word_id is not None and token not in ["[CLS]", "[SEP]", "[PAD]"]:
            result.append((token, id2label[label_id]))
    return result


# 测试预测
if __name__ == "__main__":
    text = "邓小平同志是中国改革开放的总设计师"
    print(predict(text))
