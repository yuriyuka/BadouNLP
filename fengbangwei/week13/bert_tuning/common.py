# -*- coding: utf-8 -*-
"""
@Time ： 2025/7/30 16:30
@Auth ： fengbangwei
@File ：common.py

"""
import torch
import numpy as np
import random


def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            # index = int(torch.argmax(y))
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
