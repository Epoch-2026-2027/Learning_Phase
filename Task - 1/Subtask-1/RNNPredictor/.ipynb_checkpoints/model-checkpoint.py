import torch
from torch import nn
import numpy as np

class SimpleRNNPredictor(torch.nn.Module):
    def __init__(self,history):
        super().__init__()
        self.history = history
        self.W_y = nn.Sequential(
            nn.Linear(self.history+1,64),
            nn.ReLU(),
            nn.Linear(64,10),
        )
        self.W_h = nn.Sequential(
            torch.nn.Linear(self.history+1,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,self.history),
        )

    def forward(self,x):
        h = torch.randn(len(x),self.history).to(device)
        outputs = []
        for i  in range(10):
            combined_inputs = torch.concat((h,x[:,i:i+1]),dim = 1)
            o = self.W_y(combined_inputs)
            outputs.append(o)
            h =self.store(combined_inputs)
        y = torch.concat(outputs,dim=1)
        return y

    def store(self,combined_inputs):
        return torch.tanh(self.W_h(combined_inputs))
    
