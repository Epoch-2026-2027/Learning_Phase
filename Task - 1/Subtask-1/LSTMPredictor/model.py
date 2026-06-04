import torch
from torch import nn
import numpy as np

class LSTMPredictor(torch.nn.Module):
    def __init__(self,history,cell_size,device):
        super().__init__()
        self.history = history
        self.cell_size = cell_size
        self.device = device
        self.combined_size = self.history+1
        self.W_f = nn.Sequential(
            torch.nn.Linear(self.combined_size,self.cell_size),
            torch.nn.Sigmoid(),
        )
        self.W_i1 = nn.Sequential(
            torch.nn.Linear(self.combined_size,self.cell_size),
            torch.nn.Sigmoid(),
        )
        self.W_i2 = nn.Sequential(
            torch.nn.Linear(self.combined_size,self.cell_size),
            torch.nn.Tanh(),
        )
        self.W_hc = nn.Sequential(
            nn.Linear(self.cell_size,self.history),
            nn.Tanh()
        )
        self.W_hh = nn.Sequential(
            nn.Linear(self.combined_size,self.history),
            nn.Sigmoid()
        )
        self.W_oh = nn.Sequential(
            nn.Linear(self.history*10+self.cell_size+10,100)
        )

    def forward(self,x):
        h = torch.zeros(len(x),self.history).to(self.device)
        c = torch.zeros(len(x),self.cell_size).to(self.device)
        outputs = []
        for i  in range(10):
            combined_inputs = torch.concat((h,x[:,i:i+1]),dim = 1)

            # forget part
            c= c * self.W_f(combined_inputs)
            # storing info
            c= c + self.W_i1(combined_inputs)*self.W_i2(combined_inputs)
            # getting output
            h = self.W_hh(combined_inputs) * self.W_hc(c)
            outputs.append(h)
        h = torch.concat(outputs,dim = 1)
        y = self.W_oh(torch.concat((h,c,x),dim = 1))
        return y
    
