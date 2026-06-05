import torch
import numpy as np
from torch.utils.data import Dataset,random_split

class NumArrayDataset(Dataset):
    """
    Dataset for model
    
    One-hot encoding is done for the ranks and the input as in the values are normalised
    """
    def __init__(self,data):
        self.data = np.array(data,dtype = "int")
        self.x = torch.Tensor(self.data[:,:10])
        self.y = self.data[:,10:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,i):
        self.x[i] -= self.x[i].mean()
        self.x[i] /= self.x[i].std()
        return (self.x[i],self.y[i])
        
        

def createDatasets(data,valid_ratio = .1, test_ratio = .1):
    """
    to get the separated data set and is transformed as in the class NumArrayDataset
    """
    dataset = NumArrayDataset(data)
    test_size = int(test_ratio*len(dataset))
    valid_size = int(valid_ratio*len(dataset))
    training_size = len(dataset) - test_size - valid_size
    (train_dataset, test_dataset, valid_dataset) = random_split(dataset, [training_size,test_size,valid_size])
    return (train_dataset, test_dataset, valid_dataset), (training_size, test_size, valid_size) 
    
