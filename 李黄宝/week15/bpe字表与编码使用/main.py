from model import to_bpe, bpe_to_utf8, init_vocab, up_data_vocab


# 初始化字表库 - 涵盖bpe字表生成 - 入参数(含utf-8的总字表大小)
init_vocab(500)
# 也可以更新字表库
# up_data_vocab(500)


text = '英雄技能描述: 飞行导弹攻击！'
print('======================= 测试开始 =======================')
print(f'处理的文本为: {text}')
# 过滤的标点符号，且标点符号前后内容不会参与bpe连算
bpe = to_bpe(text)
print(f'转化后的bpe为: {bpe}')
backUtf8 = bpe_to_utf8(bpe)
print(f'根据转换的bpe转回utf-8: {backUtf8}')
print("======================= 比较分割线 ")
print(f'bpeToUtf8: {backUtf8}')
print(f'utf8.    : {list(text.encode("utf-8"))}')
print('======================= 测试结束 =======================')
