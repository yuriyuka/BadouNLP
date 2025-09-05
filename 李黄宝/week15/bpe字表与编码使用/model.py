import json
import os
import re
import sys
# 前端的get set看着顺眼 - 获取字表
def getVocab():
    with open("newVocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab
# 更新字表
def setVocab(data):
    with open("newVocab.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

# 会根据符号切割文案，并且被切割的不会参与bpe连算
def get_heroes_data():
    hero_data = []
    punctuation_pattern = r'([，。！？；：,\.!\?;:]+)'
    # 获取所有文件
    for file_name in os.listdir('Heroes'):
        # 判断文件类型
        if file_name.endswith(".txt"):
            # 打开文件
            with open(os.path.join('Heroes', file_name), "r", encoding="utf-8") as file:
                intro = file.read()
                # 处理数据（采用空格区分,空格为32，根据如果有32就可以跳过）
                for line_item in intro.split("\n"):
                    for segment in re.split(punctuation_pattern, line_item):
                        if segment:
                            hero_data.append(segment + ' ')
    return hero_data
# bpe主流程，会根据32跳过
def statistical(arr, vocab):
    statistical_data = {}
    for index, item in enumerate(arr):
        if index == len(arr) - 1 or item == 32 or arr[index+1] == 32:
            continue
        key = f'{item}&{arr[index+1]}'
        if key not in vocab.values():
            statistical_data[key] = statistical_data.get(key, 0) + 1
    max_key = max(statistical_data, key=statistical_data.get)

    # return {'statistical_data':statistical_data, 'max_key':max_key}
    return max_key
# 初始化 - 会重置字表库
def init_vocab(num = 300):
    setVocab({})
    up_data_vocab(num)
# 更新字表库
def up_data_vocab(num = 300):
    utf8_data = []
    heroes_data = get_heroes_data()
    for file_name in heroes_data:
        utf8_value = list(f"{file_name}".encode("utf-8"))
        utf8_data.append(utf8_value)
    utf8_data = sum(utf8_data, [])
    vocabs = getVocab()

    for i in range(num - 256):
        max_key = statistical(utf8_data, vocabs)
        vocabs[i + 256] = max_key
        utf8_data = merge(utf8_data, max_key, i + 256)
        # 为什么这里使用sys？ 因为我用输出文件 # xx > console.txt
        print(f"当前进度： {i}/{num - 256}", file=sys.stderr)
        sys.stderr.flush()
    
    setVocab(vocabs)

# 合并数组内容
def merge(arr, pair, val): 
    newids = []
    i = 0
    while i < len(arr):
        key = f'{arr[i]}&{arr[i+1]}' if i < (len(arr) - 2)  else ''
        if key == pair:
            newids.append(val)
            i += 2
        else:
            newids.append(arr[i])
            i += 1
    return newids

# 文本转为bpe
def to_bpe(value = ''):
    vocabs = getVocab()
    reverse_vocabs = {v: k for k, v in vocabs.items()}
    tokens = list(value.encode("utf-8"))
    judge = True
    while judge:
        judge = False
        for index, item in enumerate(tokens):
            if index == len(tokens) - 1:
                break
            key = f'{item}&{tokens[index+1]}'
            if key in vocabs.values():
                tokens = merge(tokens, key, reverse_vocabs[key])
                judge = True
    return tokens

# bep转回utf-8
def bpe_to_utf8(tokens, text = ''):
    vocabs = getVocab()
    judge = True
    backUtf8 = tokens
    while judge:
        judge = False
        newTokens = []
        for index, item in enumerate(backUtf8):
            if f'{item}' in vocabs: 
                arr = vocabs[f'{item}'].split('&')
                for i in arr: newTokens.append(int(i))
                judge = True
            else:
                newTokens.append(item)
        backUtf8 = newTokens
    return backUtf8
        


