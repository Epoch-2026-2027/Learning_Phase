import torch
from torch import nn

class SimpleRNNPredictor(nn.Module):
    def __init__(self,history,device):
        super().__init__()
        self.history = history
        self.device = device
        self.W_y = nn.Sequential(
            nn.Linear(self.history+1,10*10),
        )
        self.W_h = nn.Sequential(
            nn.Linear(self.history+1,self.history),
            nn.Tanh()
        )

    def forward(self,x):
        h = torch.randn(len(x),self.history).to(self.device)
        outputs = []
        for i  in range(10):
            combined_inputs = torch.concat((h,x[:,i:i+1]),dim = -1)
            o = self.W_y(combined_inputs)
            outputs.append(o)
            h =self.store(combined_inputs)
        y = outputs[-1]
        return y

    def store(self,combined_inputs):
        return self.W_h(combined_inputs)
    
