# -*- coding: utf-8 -*-

import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NerDataset(Dataset):
    def __init__(self, file_path, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with open(config["schema_path"], 'r', encoding='utf8') as f:
            self.label_to_index = json.load(f)
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.config["class_num"] = len(self.label_to_index)
        logger.info(f"Label schema loaded. Total labels: {self.config['class_num']}")
        self.sentences = self.load_data(file_path)

    def load_data(self, file_path):
        sentences = []
        with open(file_path, 'r', encoding='utf8') as f:
            # Each sentence is a block of "char TAG" lines, separated by a blank line
            current_sentence = []
            for line in f:
                line = line.strip()
                if line:
                    if len(line.split()) != 2:
                        # Handle cases with space characters
                        parts = line.split()
                        char = ' '
                        tag = parts[-1]
                    else:
                        char, tag = line.split()
                    current_sentence.append((char, tag))
                else:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
            # Add the last sentence if the file doesn't end with a blank line
            if current_sentence:
                sentences.append(current_sentence)
        logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        chars = [s[0] for s in sentence]
        tags = [s[1] for s in sentence]

        # Tokenize and align labels
        # The input for BERT should be the string of characters
        text = "".join(chars)
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_offsets_mapping=True
        )
        
        token_labels = [-100] * self.config['max_length']  # -100 is the ignore_index for CrossEntropyLoss
        
        # `offset_mapping` gives (start, end) character indices for each token
        # We can use this to align our character-level tags with BERT's wordpiece tokens
        offsets = encoding['offset_mapping']
        
        char_index = 0
        for token_index, offset in enumerate(offsets):
            # Skip special tokens [CLS], [SEP], and padding
            if offset[0] == 0 and offset[1] == 0:
                continue

            # Assign label to the first sub-token of a character
            if char_index < len(tags):
                # The start of the token offset should match the current character position
                # This is a simplified alignment that works well for character-based tokenization
                token_labels[token_index] = self.label_to_index.get(tags[char_index], self.label_to_index['O'])
                # If a character is split into multiple tokens, subsequent tokens get -100
                # Move to the next character only when we've processed all tokens for the current one
                if offset[1] > char_index + 1:
                     char_index += 1
            
        input_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(token_labels)
        
        return input_ids, attention_mask, labels

def load_data(data_path, config, shuffle=True):
    tokenizer = BertTokenizerFast.from_pretrained(config["pretrain_model_path"])
    dataset = NerDataset(data_path, config, tokenizer)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)