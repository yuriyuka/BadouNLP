from transformers import AutoModelForTokenClassification
from config import Config
from torch.optim import Adam, SGD

# 使用TokenClassification模型而不是SequenceClassification
TorchModel = AutoModelForTokenClassification.from_pretrained(
    Config["pretrain_model_path"],
    num_labels=Config["num_labels"]
)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
