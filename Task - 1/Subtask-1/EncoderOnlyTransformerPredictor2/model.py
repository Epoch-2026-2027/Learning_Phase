import torch
from torch import mean, nn
import numpy as np


class AddAndNormalise(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))


    def forward(self,x,y):
        h = torch.concat([x,y],dim = -1)
        return self.a*(h - torch.mean(h,dim = -1,keepdim=True))/torch.std(h,dim=-1,keepdim=True) + self.b


class MultiHeadAttention(nn.Module):
    def __init__(self,nheads,input_dim,attention_layers,device):
        super().__init__()
        self.device = device
        self.W_Q = nn.Parameter(torch.randn((input_dim,attention_layers*nheads))/input_dim**.5)
        self.W_K = nn.Parameter(torch.randn((input_dim,attention_layers*nheads))/input_dim**.5)
        self.W_V = nn.Parameter(torch.randn((input_dim,attention_layers*nheads))/input_dim**.5)
        self.W_O = nn.Parameter(torch.randn((attention_layers*nheads,attention_layers*nheads))/(attention_layers)**.5)

        self.sf = nn.Softmax(dim=-1)
        self.attention_layers = attention_layers
        self.nheads = nheads

    def forward(self,x):
        batchsize, input_len = x.shape[0],x.shape[1]
        Q = x@self.W_Q
        K = x@self.W_K
        V = x@self.W_V

        """
        previous attempt
        Qs = torch.chunk(Q,self.nheads,dim = -1)
        Ks = torch.chunk(K,self.nheads,dim = -1)
        Vs = torch.chunk(V,self.nheads,dim = -1)
        c = torch.concat([self.scaledDotProduct(Qs[i],Ks[i],Vs[i]) for i in range(self.nheads)],dim=-1)
        return c@self.W_O
        """
        
        Qs = Q.reshape(batchsize,input_len,self.nheads,self.attention_layers).transpose(1,2)
        Ks = K.reshape(batchsize,input_len,self.nheads,self.attention_layers).transpose(1,2)
        Vs = V.reshape(batchsize,input_len,self.nheads,self.attention_layers).transpose(1,2)

        out = self.scaledDotProduct(Qs,Ks,Vs).transpose(1,2).reshape(batchsize,input_len,-1)
        return out@self.W_O
        

    def scaledDotProduct(self,Q,K,V):
        return self.sf(1/torch.sqrt(torch.Tensor([Q.shape[-1]]).to(self.device))*(Q@(K.transpose(-1,-2))))@V


class EncoderOnlyTransformerPredictor2(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.att = MultiHeadAttention(4,2,16,device)
        self.norm = AddAndNormalise()
        self.W_1 = nn.Parameter(torch.randn((4*16+2,64))/66**0.5)
        self.b_1 = nn.Parameter(torch.randn(64)*0.1)
        self.W_2 = nn.Parameter(torch.randn((64,10))/8)
        self.b_2 = nn.Parameter(torch.randn(10)*0.1)
        self.r = nn.ReLU()
        self.ff = lambda x : self.r(x@self.W_1+self.b_1)@self.W_2 + self.b_2

    def forward(self,x):
        x2 = self.position_embedded(x)
        x3 = self.att(x2)
        x4 = self.norm(x2,x3)
        
        x5 = self.ff(x4)
        return x5

    def position_embedded(self,x):
        x = x.reshape(-1,x.shape[1],1)
        P = torch.zeros_like(x)
        for i in range(x.shape[1]):
            P[:,i,0] = (1.5)**(-i)
            #P[:,i,0] = np.sin(i)
        return torch.concat([x,P], dim = -1)
