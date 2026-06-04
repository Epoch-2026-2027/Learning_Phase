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

class EncoderOnlyTransformerPredictor(torch.nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.att = Attention(2,128)
        self.norm = AddAndNormalise()
        self.W_1 = nn.Parameter(torch.randn((128+2,64)))
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