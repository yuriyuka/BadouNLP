# -*- coding: utf-8 -*-
import torch
import json
from loader import load_data
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2 # Use IOB2 scheme for evaluation

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        
        # Load schema to map indices back to labels
        with open(config["schema_path"], 'r', encoding='utf8') as f:
            self.label_to_index = json.load(f)
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

    def eval(self, epoch):
        self.logger.info(f"Starting evaluation for epoch {epoch}...")
        self.model.eval()
        
        all_pred_labels = []
        all_true_labels = []

        with torch.no_grad():
            for batch_data in self.valid_data:
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]
                
                input_ids, attention_mask, labels = batch_data
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                for i in range(labels.shape[0]):
                    # Remove padding and special tokens for evaluation
                    true_seq = [self.index_to_label[l.item()] for l in labels[i] if l != -100]
                    pred_seq = [self.index_to_label[p.item()] for p, l in zip(predictions[i], labels[i]) if l != -100]
                    
                    all_true_labels.append(true_seq)
                    all_pred_labels.append(pred_seq)

        self.show_stats(all_true_labels, all_pred_labels)

    def show_stats(self, true_labels, pred_labels):
        # Using seqeval's classification_report
        report = classification_report(true_labels, pred_labels, scheme=IOB2, mode='strict', digits=4)
        self.logger.info("NER Classification Report:")
        self.logger.info("\n" + report)
        self.logger.info("--------------------")