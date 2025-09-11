'''
BPE算法实现
'''
import torch
import numpy as np
from collections import defaultdict

def string_to_unicode8(string):
    tokens=string.encode('utf-8', errors="replace")
    tokens=list(map(int,tokens))
    return tokens#一个列表，字符转换为整数后的列表

def get_pair_freq(int_ids):

    pair_freq= {}
    for word in zip(int_ids,int_ids[1:]):
        pair_freq[word]=pair_freq.get(word,0)+1
    return pair_freq

def get_max_paire_freq(pair_freq):
    max_pair_freq=max(pair_freq,key=pair_freq.get)
    return max_pair_freq

def replace_update_tokens(tokens,max_pair_freq,number):
    new_tokens=[]
    #为啥不用for循环，因为这里面不好变动i的值，继承到下一次循环，for i in range(len(tokens)-1):
    i=0
    while i<len(tokens):
        if  i+1<len(tokens) and (tokens[i],tokens[i+1])==max_pair_freq:
            new_tokens.append(number)
            i+=2
        else:
            new_tokens.append(tokens[i])
            i+=1
    return new_tokens

def bpe_encode_string(string,times,old_vocab_size):
    tokens=string_to_unicode8(string)
    merge={}
    for i in range(times):
        pair_freq=get_pair_freq(tokens)
        max_pair_freq=get_max_paire_freq(pair_freq)
        tokens=replace_update_tokens(tokens,max_pair_freq,old_vocab_size+i)
        merge[max_pair_freq]=old_vocab_size+i
    return tokens,merge

def get_vocab_for_new_tokens_to_bytes(merge):
    vocab={int_idx:bytes([int_idx]) for int_idx in range(256)}
    for (pair0,pair1),idx in merge.items():
        vocab[idx]=vocab[pair0]+vocab[pair1]#b'aa'+b'bb'=b'abbb'
    return vocab

def bpe_decode_tokens(tokens,vocab):
    bytes_saq=b"".join(vocab[idx] for idx in tokens)
    str_tar=bytes_saq.decode("utf-8", errors="replace")
    return str_tar

if __name__=="__main__":
    string='你好黑黑你好'
    old_tokens=string_to_unicode8(string)
    new_tokens,merge=bpe_encode_string(string,2,256)
    print('压缩前的unicode8编码为：\n',old_tokens)
    print('--------------------')
    print('压缩后的编码为：\n',new_tokens)
    print('--------------------')
    vocab=get_vocab_for_new_tokens_to_bytes(merge)
    print('压缩的词表为：\n',merge)
    print('--------------------')
    tar_test=list(list(merge.keys())[1])#[228, 189, 160, 229, 165, 189]
    print('测试压缩后的编码：\n',tar_test)
    print('--------------------')
    str_tar=bpe_decode_tokens(tar_test,vocab)
    print('解码后的字符串为：\n',str_tar)








