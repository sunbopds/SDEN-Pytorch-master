import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy           # copy模块包括创建复合对象(包括列表、元组、字典和用户定义对象的实例)的深浅复制的函数。
import random
from tqdm import tqdm               # Tqdm 是一个快速，可扩展的Python进度条，
flatten = lambda l: [item for sublist in l for item in sublist]     #


def prepare_dataset(path,built_vocab=None,user_only=False):     # 准备数据集
    data = open(path,"r",encoding="utf-8").readlines()          # 打开path中的文件
    p_data=[]                                                   #
    history=[["<null>"]]                                        #
    for d in data:                                              # 遍历data
        if d=="\n":                                             # 如果句子等于\n,则运行下面句子
            history=[["<null>"]]
            continue
        dd = d.replace("\n","").split("|||")                    # 去掉\n,用|||分割
        if len(dd)==1:                                          # 如果dd长度为1
            if user_only:                                       # 如果user_only为真，默认为假
                pass                                            # 跳过本轮循环
            else:
                bot = dd[0].split()                             # 切分字符串
                history.append(bot)                             # 加入history
        else:
            user = dd[0].split()                                # dd[0]切分出user
            tag = dd[1].split()                                 # dd[1]切分出tag
            intent = dd[2]
            temp = deepcopy(history)                            # 深层copy
            p_data.append([temp,user,tag,intent])               # 形成list，加入p_data中
            history.append(user)                                # 只把user加入history
    
    if built_vocab is None:                                     # 如果build_vocab为None
        historys, currents, slots, intents = list(zip(*p_data)) # 将元素打包成一个元祖，返回列表
        vocab = list(set(flatten(currents)))                    # flatten将多层list，压缩成1层，set()无序不重复元素集,
        slot_vocab = list(set(flatten(slots)))                  #
        intent_vocab = list(set(intents))
        
        word2index={"<pad>" : 0, "<unk>" : 1, "<null>" : 2, "<s>" : 3, "</s>" : 4}   # 词的index
        for vo in vocab:                                        #
            if word2index.get(vo)==None:                        # vo词不再word2index内
                word2index[vo] = len(word2index)                # word2index[vo]

        slot2index={"<pad>" : 0}                                # slot的索引
        for vo in slot_vocab:                                   #
            if slot2index.get(vo)==None:
                slot2index[vo] = len(slot2index)

        intent2index={}
        for vo in intent_vocab:
            if intent2index.get(vo)==None:
                intent2index[vo] = len(intent2index)
    else:
        word2index, slot2index, intent2index = built_vocab
        
    for t in tqdm(p_data):                                      # 显示进度条
        for i,history in enumerate(t[0]):                       # 枚举t[0]
            t[0][i] = prepare_sequence(history, word2index).view(1, -1) # prepare_sequence函数

        t[1] = prepare_sequence(t[1], word2index).view(1, -1)   #
        t[2] = prepare_sequence(t[2], slot2index).view(1, -1)
        t[3] = torch.LongTensor([intent2index[t[3]]]).view(1,-1)
            
    if built_vocab is None:                                     # 如果built_vocab为None，则返回下面的
        return p_data, word2index, slot2index, intent2index
    else:                                                       # 否则返回p_data
        return p_data

def prepare_sequence(seq, to_index):                            # 准备序列
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<unk>"], seq))
    # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    # add = lambda x, y : x+y     # add(1,2)  # 结果为3
    #
    return torch.LongTensor(idxs)

def data_loader(train_data,batch_size,shuffle=False):   # 迭代函数，每次运行输出batch
    if shuffle: random.shuffle(train_data)              # 如果shuffle为True，shuffle()方法将序列的所有元素随机排序
    sindex = 0                                          # sindex是起始的索引号，eindex是末尾的索引号
    eindex = batch_size                                 # batch_size 赋值给eindex
    while eindex < len(train_data):                     # 如果eindex < train_data的长度，循环，否则结束循环
        batch = train_data[sindex: eindex]              # 取出train_data从0到eindex的数据，给batch
        temp = eindex                                   # 赋值给temp
        eindex = eindex + batch_size                    # eindex扩大batch_size大小
        sindex = temp                                   # temp赋值给sindex
        yield batch
    
    if eindex >= len(train_data):                       # eindex最后一个，把剩余的数据batch出来
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch, w_to_ix,s_to_ix): # for bAbI dataset
    history,current,slot,intent = list(zip(*batch))
    max_history = max([len(h) for h in history])
    max_len = max([h.size(1) for h in flatten(history)])
    max_current = max([c.size(1) for c in current])
    max_slot = max([s.size(1) for s in slot])
    
    historys, currents, slots = [], [], []
    for i in range(len(batch)):
        history_p_t = []
        for j in range(len(history[i])):
            if history[i][j].size(1) < max_len:
                history_p_t.append(torch.cat([history[i][j], torch.LongTensor([w_to_ix['<pad>']] * (max_len - history[i][j].size(1))).view(1, -1)], 1))
            else:
                history_p_t.append(history[i][j])

        while len(history_p_t) < max_history:
            history_p_t.append(torch.LongTensor([w_to_ix['<pad>']] * max_len).view(1, -1))

        history_p_t = torch.cat(history_p_t)
        historys.append(history_p_t)

        if current[i].size(1) < max_current:
            currents.append(torch.cat([current[i], torch.LongTensor([w_to_ix['<pad>']] * (max_current - current[i].size(1))).view(1, -1)], 1))
        else:
            currents.append(current[i])

        if slot[i].size(1) < max_slot:
            slots.append(torch.cat([slot[i], torch.LongTensor([s_to_ix['<pad>']] * (max_slot - slot[i].size(1))).view(1, -1)], 1))
        else:
            slots.append(slot[i])

    currents = torch.cat(currents)
    slots = torch.cat(slots)
    intents = torch.cat(intent)
    
    return historys, currents, slots, intents

def pad_to_history(history, x_to_ix): # this is for inference
    
    max_x = max([len(s) for s in history])
    x_p = []
    for i in range(len(history)):
        h = prepare_sequence(history[i],x_to_ix).unsqueeze(0)
        if len(history[i]) < max_x:
            x_p.append(torch.cat([h,torch.LongTensor([x_to_ix['<pad>']] * (max_x - h.size(1))).view(1, -1)], 1))
        else:
            x_p.append(h)
        
    history = torch.cat(x_p)
    return [history]