import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class SDEN(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,slot_size,intent_size,dropout=0.3,pad_idx=0):
        super(SDEN,self).__init__()
        
        self.pad_idx = 0
        self.embed = nn.Embedding(vocab_size,embed_size,padding_idx=self.pad_idx)           # 随机初始化一个矩阵，矩阵的长是vocab_size, 宽是 embedd_size
        self.bigru_m = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)   # 门循环单元
        self.bigru_c = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)   #
        self.context_encoder = nn.Sequential(nn.Linear(hidden_size*4,hidden_size*2),        # Sequential 是一个容器，模块按照构造函数中传递的顺序添加到模块中
                                                               nn.Sigmoid())                # nn.Linear是一种线性变换
        self.session_encoder = nn.GRU(hidden_size*2,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.decoder_1 = nn.GRU(embed_size,hidden_size*2,batch_first=True,bidirectional=True)
        self.decoder_2 = nn.LSTM(hidden_size*4,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.intent_linear = nn.Linear(hidden_size*4,intent_size)
        self.slot_linear = nn.Linear(hidden_size*4,slot_size)
        self.dropout = nn.Dropout(dropout)

        for param in self.parameters():
            if len(param.size())>1:
                nn.init.xavier_uniform_(param)                                          # 初始化参数
            else:
                param.data.zero_()                                                      # 否则参数为0
        
    def forward(self,history,current):
        batch_size = len(history)
        H= [] # encoded history
        for h in history:
            mask = (h!=self.pad_idx)                                                    # mask存储值历史信息h和pad_idx比较后，true or false
            length = mask.sum(1).long()                                                 # 计算mask的长度
            embeds = self.embed(h)                                                      # 对h求向量
            embeds = self.dropout(embeds)                                               # 对embeds向量做dropout
            lens, indices = torch.sort(length, 0, True)                                 # 对length进行排序
            lens = [l if l>0 else 1 for l in lens.tolist()] # all zero-input            # lens转list，如果l大于0，输出l，否则输出0
            packed_h = pack(embeds[indices], lens, batch_first=True)                    # 将embeds，lens打包
            outputs, hidden = self.bigru_m(packed_h)                                    # 对packed_h进行bigru
            _, _indices = torch.sort(indices, 0)                                        # 排序indices
            hidden = torch.cat([hh for hh in hidden],-1)                                # 向量拼接
            hidden = hidden[_indices].unsqueeze(0)                                      # unsqueeze把向量转为一列
            H.append(hidden)                                                            # 加入H中
        
        M = torch.cat(H) # B,T_C,2H                                                     # 拼接H为一个M向量
        M = self.dropout(M)                                                             # 对M进行dropout
        
        embeds = self.embed(current)                                                    # 对当前对话进行embed
        embeds = self.dropout(embeds)                                                   # 对当前对话dropout
        mask = (current!=self.pad_idx)                                                  # mask进行比较true or false
        length = mask.sum(1).long()                                                     # 求长度
        lens, indices = torch.sort(length, 0, True)                                     # 排序
        packed_h = pack(embeds[indices], lens.tolist(), batch_first=True)               # 打包
        outputs, hidden = self.bigru_c(packed_h)                                        # bigru
        _, _indices = torch.sort(indices, 0)                                            # 排序
        hidden = torch.cat([hh for hh in hidden],-1)                                    # 拼接
        C = hidden[_indices].unsqueeze(1) # B,1,2H                                      # 转为一列
        C = self.dropout(C)                                                             # dropout
        
        C = C.repeat(1,M.size(1),1)                                                     # repeat是沿着指定的维度重复tensor
        CONCAT = torch.cat([M,C],-1) # B,T_c,4H                                         # 拼接M和C
        
        G = self.context_encoder(CONCAT)                                                # 放入Sequential容器中
        
        _,H = self.session_encoder(G) # 2,B,2H                                          # 进行GRU
        weight = next(self.parameters())                                                # 跳到下一个参数
        cell_state = weight.new_zeros(H.size())                                         #
        O_1,_ = self.decoder_1(embeds)                                                  # 解码输出第1层O_1
        O_1 = self.dropout(O_1)                                                         # dropout输出第1层O_1
        
        O_2,(S_2,_) = self.decoder_2(O_1,(H,cell_state))                                # 第二层解码输出O_2
        O_2 = self.dropout(O_2)                                                         # dropout O_2 第二层的输出
        S = torch.cat([s for s in S_2],1)                                               # 拼接S_2,获得S
        
        intent_prob = self.intent_linear(S)                                             # 线性函数获取intent
        slot_prob = self.slot_linear(O_2.contiguous().view(O_2.size(0)*O_2.size(1),-1)) # 获取slot
        
        return slot_prob, intent_prob