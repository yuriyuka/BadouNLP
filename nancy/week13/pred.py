# -*- coding: utf-8 -*-
import torch
import logging
from model import build_base_model
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from evaluate import Evaluator
from config import Config

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 大模型微调策略
tuning_tactics = Config["tuning_tactics"]
print("正在使用 %s" % tuning_tactics)

if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "q_proj", "k_proj", "v_proj", "o_proj", "dense", "out_proj"],
        task_type="TOKEN_CLS",
    )
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
else:
    raise ValueError("未知的tuning_tactics: %s" % tuning_tactics)

# 重建模型
base_model = build_base_model()
model = get_peft_model(base_model, peft_config)

state_dict = model.state_dict()

# 将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('output/lora_tuning.pth', map_location='cpu')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('output/p_tuning.pth', map_location='cpu')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('output/prompt_tuning.pth', map_location='cpu')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('output/prefix_tuning.pth', map_location='cpu')

print(loaded_weight.keys())
state_dict.update(loaded_weight)

# 权重更新后重新加载到模型
model.load_state_dict(state_dict)

# 进行一次评估
model = model.to(device)
evaluator = Evaluator(Config, model, logger)
evaluator.eval(0)


