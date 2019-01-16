import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim                     # 优化算法包
import numpy as np
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics            # 机器学习数据集和学习方法
import argparse                                 # 解析命令

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 判断是CPU还是GPU


def evaluation(model,dev_data):
    model.eval()
    index2slot = {v:k for k,v in model.slot_vocab.items()}
    preds=[]
    labels=[]
    hits=0
    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data,32,True)):
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h,c)

            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            hits+=torch.eq(intent_p.max(1)[1],intent.view(-1)).sum().item()


    print(hits/len(dev_data))
    
    sorted_labels = sorted(
    list(set(labels) - {'O','<pad>'}),
    key=lambda name: (name[1:], name[0])
    )
    
    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds] 
    labels = [[y] for y in labels]
    
    print(metrics.flat_classification_report(
    labels, preds, labels = sorted_labels, digits=3
    ))

def save(model,config):
    checkpoint = {
                'model': model.state_dict(),
                'vocab': model.vocab,
                'slot_vocab' : model.slot_vocab,
                'intent_vocab' : model.intent_vocab,
                'config' : config,
            }
    torch.save(checkpoint,config.save_path)
    print("Model saved!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()                                  # 获得参数，并创建解析对象
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')            # 参数--mode，默认train
    parser.add_argument('--pause', type=int, default=0)                 # 参数pause
    parser.add_argument('--iteration', type=str, default='0')           # 迭代
    parser.add_argument('--epochs', type=int, default=5,                # 训练轮数
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=64,           # 批次大小
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,              # 学习率
                        help='learning_rate')
    parser.add_argument('--dropout', type=float, default=0.3,           # droppout
                        help='dropout')
    parser.add_argument('--embed_size', type=int, default=100,          # embed大小
                        help='embed_size')
    parser.add_argument('--hidden_size', type=int, default=64,          # 隐层大小
                        help='hidden_size')
    parser.add_argument('--save_path', type=str, default='weight/model.pkl',    # 模型保持路径
                        help='save_path')
    
    config = parser.parse_args()                                       # 配置保持到config
    
    train_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob')    # 预读数据
    # train_data是tensor，例子：tensor([[ 715,  325,  940, 1174,  366,  586,  309, 1166,  762]]),
    # word2index是元祖，例子：{'assitant': 5, 'people': 6, 'again': 7, 'got': 8
    # slot2index: {'I-room': 13, '<pad>': 0, 'I-agenda': 1, 'B-date': 2, 'B-distance': 15,
    # intent2index: {'navigate': 0, 'thanks': 1, 'schedule': 2, 'weather': 3} 只有3个intent
    dev_data = prepare_dataset('data/dev.iob',(word2index,slot2index,intent2index))         # 预读数据
    # dev_data: tensor([[ 376, 1170,  633, 1146,  918]]), tensor

    model = SDEN(len(word2index),config.embed_size,config.hidden_size,\
                 len(slot2index),len(intent2index),word2index['<pad>'])                 # SDEN类，初始化,
    # len(word2index): 1179
    # config.embed_size: 100
    # config.hidden_size: 64
    # len(slot2index): 24
    # len(intent2index): 4
    # word2index['<pad>'] = 0, <pad>字符的索引号

    model.to(device)                                # 设置模型对应设备
    model.vocab = word2index                        # 模型的词索引
    model.slot_vocab = slot2index                   # 模型的槽索引
    model.intent_vocab = intent2index               # 模型的意图索引
    
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)    # loss函数，采用交叉熵
    intent_loss_function = nn.CrossEntropyLoss()                # intent，采用交叉熵
    optimizer = optim.Adam(model.parameters(),lr=config.lr)     # 优化器Adam
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,milestones=[config.epochs//4,config.epochs//2],optimizer=optimizer)
    
    model.train()   # 模型训练模式
    file = open('predict.txt', 'w')
    for epoch in range(config.epochs):          # 开始训练
        losses=[]
        scheduler.step()        # 调度步
        for i, batch in enumerate(data_loader(train_data, config.batch_size, True)):   # 调用data_utils中的datat_loader函数
            h, c, slot, intent = pad_to_batch(batch, model.vocab, model.slot_vocab)  # data_utils的pad_to_batch函数
            h = [hh.to(device) for hh in h]             # h是history
            c = c.to(device)                            # c是current
            slot = slot.to(device)
            intent = intent.to(device)
            model.zero_grad()                           # 把模型的梯度设为0
            slot_p, intent_p = model(h,c)               # 调用model中的forward，返回slot和intent的预测值
            file.write(str(slot_p) + "————" + str(intent_p))

            loss_s = slot_loss_function(slot_p,slot.view(-1))           # slot的loss_function
            loss_i = intent_loss_function(intent_p,intent.view(-1))     # intent的loss_function
            loss = loss_s + loss_i                                      # 总loss是slot和intent的loss相加
            losses.append(loss.item())                                  # 加入losses中
            loss.backward()                                             # 向后求导

            optimizer.step()                                            # 优化
            if i % 100 == 0:                                            # 每100轮
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch,config.epochs,i,len(train_data)//config.batch_size,np.mean(losses)))   # 打印平均loss
                losses=[]

    file.close()
    evaluation(model,dev_data)  # 评估模型
    save(model,config)          # 保存模型