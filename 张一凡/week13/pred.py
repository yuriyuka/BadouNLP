import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig
from evaluate import Evaluator
from config import Config

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tuning_tactics = Config["tuning_tactics"]
print("正在使用 %s" % tuning_tactics)

# LoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
    task_type="TOKEN_CLS"
)

# 重建模型
model = TorchModel
model = get_peft_model(model, peft_config)

# 加载微调权重
loaded_weight = torch.load(f'output/{tuning_tactics}.pth')
model.load_state_dict(loaded_weight, strict=False)

# 进行一次测试
model = model.cuda()
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)
