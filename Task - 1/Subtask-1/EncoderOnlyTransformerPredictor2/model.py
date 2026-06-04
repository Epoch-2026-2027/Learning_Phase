import torch
from torch import nn
import numpy as np

class Attention(nn.Module):
    def __init__(self,size,size2):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn((size,size2)))
        self.W_K = nn.Parameter(torch.randn((size,size2)))
        self.W_V = nn.Parameter(torch.randn((size,size2)))
        self.sf = nn.Softmax(dim=-1)
        self.size2 = size2

    def forward(self,x):
        return self.sf(1/np.sqrt(self.size2)*((x@self.W_Q)@(x@self.W_V).mT))@x@self.W_V

class AddAndNormalise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,y):
        h = torch.concat([x,y],dim = -1)
        return (h - h.mean())/h.std()


class MultiHeadAttention(nn.Module):
    def __init__(self,nheads,input_dim,attention_layers,device):
        super().__init__()
        self.device = device
        self.W_Q = nn.Parameter(torch.randn((input_dim,attention_layers*nheads)))
        self.W_K = nn.Parameter(torch.randn((input_dim,attention_layers*nheads)))
        self.W_V = nn.Parameter(torch.randn((input_dim,attention_layers*nheads)))
        self.W_O = nn.Parameter(torch.randn((attention_layers*nheads,attention_layers*nheads)))
        
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
        return self.sf(1/torch.sqrt(torch.Tensor([Q.shape[-1]]).to(self.device))*(Q@(K.mT)))@V


class EncoderOnlyTransformerPredictor2(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.att = MultiHeadAttention(4,2,16,device)
        self.norm = AddAndNormalise()
        self.W_1 = nn.Parameter(torch.randn((4*16+2,64)))
        self.b_1 = nn.Parameter(torch.randn(64))
        self.W_2 = nn.Parameter(torch.randn((64,10)))
        self.b_2 = nn.Parameter(torch.randn(10))
        self.r = nn.ReLU()
        self.ff = lambda x : self.r(x@self.W_1+self.b_1)@self.W_2 + self.b_2

    def forward(self,x):
        x2 = self.position_embedded(x)
        x3 = self.att(x2)
        #print(x3.shape)
        x4 = self.norm(x2,x3)
        
        return self.ff(x4).reshape(-1,100)

    def position_embedded(self,x):
        x = x.reshape(-1,x.shape[1],1)
        P = torch.zeros_like(x)
        for i in range(x.shape[1]):
            P[:,i,0] = 2**(-i)
        return torch.concat([x,P], dim = -1)
