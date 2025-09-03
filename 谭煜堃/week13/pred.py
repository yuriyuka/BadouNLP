import torch
import logging
from model import TorchModel
from peft import get_peft_model, LoraConfig, PeftModel
from evaluate import Evaluator
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict(config):
    # Load the base model
    model = TorchModel
    
    # Load the LoRA adapter weights
    model_path = f"output/{config['tuning_tactics']}.pth"
    logger.info(f"Loading trained LoRA weights from {model_path}")
    
    # Load the entire model state_dict which includes LoRA weights
    model.load_state_dict(torch.load(model_path))

    # To use PEFT model for inference, you can merge the LoRA layers
    # model = model.merge_and_unload() # Optional: for faster inference

    if torch.cuda.is_available():
        model = model.cuda()
    
    logger.info("Model loaded successfully. Starting prediction.")
    
    # Use the evaluator to run prediction on the validation set
    evaluator = Evaluator(config, model, logger)
    evaluator.eval(epoch="final_prediction")

if __name__ == "__main__":
    predict(Config)