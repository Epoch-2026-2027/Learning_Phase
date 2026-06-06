import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

enableEmbeds = True
h=3
dmodel=9
dk=3
dv=3

X = torch.tensor([[266,630,540,560,470,561,323,720,11,461],
                [177,289,184,553,548,409,483,507,558,491],
                [214,691,636,903,582,490,389,528,555,33],
                [527,547,865,589,141,645,761,339,861,564],
                [270,553,137,275,217,321,205,507,845,438],
                [695,934,391,221,990,139,720,429,111,268],
                [239,436,217,46,500,79,865,992,539,251],
                [947,468,936,20,806,108,576,848,110,496],
                [39,200,355,119,335,382,170,618,168,407],
                [922,82,896,298,332,977,566,553,800,883],
                [966,880,63,947,193,328,39,278,902,325],
                [735,724,658,651,487,109,722,146,648,402],
                [348,662,21,90,880,676,225,510,846,69],
                [942,954,392,870,702,473,313,931,156,394]])
# print(X, X.shape, '\n')
# X = X.view(-1,dmodel)
# print(X, X.shape, '\n')
#X = X.view(h, -1, dmodel//h)#.transpose(-1,-2)
# X = torch.tensor(np.array([X[:,j:j+dmodel//h] for j in range(0, dmodel, dmodel//h)]), dtype=torch.float)
# print(X.shape, '\n') # 3,8,3

# raise
# Implementing modules for self-attention and feed-forward nn mechanisms
class SingleHeadedSelfAttention(nn.Module):
    def __init__(self, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.softmax = nn.Softmax(dim=0)
        self.Wq = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wk = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wv = nn.Parameter(torch.randn((dmodel, dv)))
        self.Wf = nn.Parameter(torch.randn((dv, dmodel)))

    def forward(self, X):
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        score = self.softmax(torch.matmul(Q,K.transpose(-2, -1))/ np.sqrt(self.dk))
        attention = torch.matmul(score,V)
        out = torch.matmul(attention,self.Wf)
        return out
    
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h
        self.softmax = nn.Softmax(dim=2)
        self.Wq = nn.Parameter(torch.randn(dmodel*dk).view(h, dmodel//h, dk))
        self.Wk = nn.Parameter(torch.randn(dmodel*dk).view(h, dmodel//h, dk))
        self.Wv = nn.Parameter(torch.randn(dmodel*dv).view(h, dmodel//h, dv))
        self.Wf = nn.Parameter(torch.randn(dmodel*dv*h).view(dv*h, dmodel))

    def forward(self, X):
        print(X.shape,'\n\n')
        X = torch.stack(torch.split(X, self.h, dim=2), dim=1)
        print(X.shape)
        # raise "the hell bro"
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        batchscore = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(self.dk), dim=2)
        headattention = torch.matmul(batchscore,V)
        # headattention.shape # batchsize, heads, tokens, dmodel//heads
        attention = torch.flatten(torch.permute(headattention, (0,2,1,3)), start_dim=2, end_dim=3)
        print(attention.shape)
        out = torch.matmul(attention,self.Wf)
        
        # print("headattention:",headattention[0,:,0],headattention.shape, '\n\n\n', "attention:", attention[0][0],attention.shape, '\n\n\noutput:',out[0][0],out.shape)
        # raise
        return out



class FFNN(nn.Module):
    def __init__(self, dmodel=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        no_input = dmodel if enableEmbeds or dmodel is not None else 1
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_input, 64),       
                                nn.ReLU(),
                                nn.Linear(64, no_input),           
                            )
    
    def forward(self, input):
        out = self.mlp_seq(input)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadatt = MultiHeadedSelfAttention(h, dmodel, dk, dv)
        self.ffnn = FFNN(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
    
    def forward(self, input):
        x1 = self.multiheadatt(input)
        print(x1.shape)
        print(input.shape)
        x1 = self.norm1(x1+input)
        x2 = self.ffnn(x1)
        out = self.norm2(x2+x1)
        return out


# Defining the Transformer-Encoder based model.
class BERTModel(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, no_num=1000, len_no=10, no_labels=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_layers = []
        no_inputs=len_no
        if enableEmbeds:
            self.embed = nn.Embedding(no_num, dmodel)
            no_inputs*=dmodel
        for _ in range(N):
            self.encoder_layers.append(EncoderLayer(h, dmodel, dk, dv))
        self.flatten = nn.Flatten()
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_inputs, 128),       
                                nn.ReLU(),
                                nn.Linear(128, 128),           
                                nn.ReLU(),
                                nn.Linear(128, 10*no_labels),  
                                nn.Tanh()
                            )
    
    def forward(self, input):
        x=self.embed(input)
        for layer in self.encoder_layers:
            x=layer(x)
        x = self.flatten(x)
        x = self.mlp_seq(x)
        return x


bert = BERTModel(N=4, h=h, dmodel=dmodel, dk=dk, dv=dv, len_no=10, no_labels=10)

# layer = EncoderLayer(h=h, dmodel=dmodel, dk=dk, dv=dv)

# mhsa = MultiHeadedSelfAttention(h=h, dmodel=dmodel, dk=dk, dv=dv)

# res1 = mhsa(X)
# print(res1, res1.shape)

print(X.shape)
res2 = bert(X)
print(res2, res2.shape)