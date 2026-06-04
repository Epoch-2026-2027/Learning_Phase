import torch
from torch import nn
import numpy as np

class MLPPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,100)
        )

    def forward(self,x):
        return self.model(x)
    
