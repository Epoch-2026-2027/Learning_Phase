import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Creating the Dataset class
class RankingDataset(Dataset):
    def __init__(self, train_dat:pd.DataFrame):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cols, lab_cols = train_dat.keys()[:10], train_dat.keys()[10:] # Retrieving the column_names for 1. numerical values  2. label values
        self.nums = torch.tensor(train_dat[num_cols].to_numpy()).to(device=device, dtype=torch.float) # Numerical values
        self.labs = torch.tensor(train_dat[lab_cols].to_numpy()).to(device=device, dtype=torch.float) # Rank values

    def __len__(self):
        return len(self.nums)
    
    def __getitem__(self, idx):
        return (self.nums[idx], self.labs[idx])

class ScalarMultiply(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        # Multiplies input tensor x by the scalar factor
        return x * self.factor


# Iteration-1 Defining the MLP based model.
class MLPModel(nn.Module):
    def __init__(self, no_num=10, no_label=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_num, no_label),       # input layer (numbers)
                                nn.ReLU(),
                                # nn.Linear(100, 50),           # hidden layer (1)
                                # nn.ReLU(),
                                # nn.Linear(25, 15),           # hidden layer (1)
                                # nn.ReLU(),
                                # nn.Linear(15, no_label),     # outpiut later (scores)
                                # nn.ReLU()
                                #ScalarMultiply(9.0)
                            )
    
    def forward(self, nums): 
        return self.mlp_seq(nums)




def MLPtrain(model, tr_dataloader, val_dataloader, epochs, l_rate, patience=10): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Moving to cuda accelerator if possible
    model.to(device)
    model.train()
    loss_fn = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=l_rate) # Stochastic Gradient Descent
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate) # Adaptive Moment Estimation, weight_decay=1e-4
    losses = []
    val_losses = []
    pat_count = 0
    epoch=0
    while epoch<epochs: # Gradient descent over epochs
        total_loss = 0
        val_loss = 0
        i=0
        for nums, labels in tr_dataloader: # Retrieving features for training
            pred = model.forward(nums)
            loss = loss_fn(pred, labels)
            total_loss += loss.item()

            loss.backward() # beginning backprop
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        
        total_loss/=i
        i=0
        
        for nums, labels in val_dataloader:
            pred = model.forward(nums)
            loss = loss_fn(pred, labels)
            val_loss += loss.item()

            loss.backward() # beginning backprop
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        
        val_loss/=i


        if losses!=[] and total_loss>losses[-1]:
            if pat_count==patience:
                break
            else:
                pat_count+=1
        else:
            pat_count=0
        losses.append(total_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses




def test(model, test_data:pd.DataFrame):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_error = 0
    i=0
    with torch.no_grad():
        for dat in test_data.itertuples():     # Retrieving the column_names for 1. numerical values  2. rank values
            dat = list(dat)[1:]
            nums = torch.tensor(dat[:10]).to(device=device, dtype=torch.float) # Numerical values
            labs = torch.tensor(dat[10:]).to(device=device, dtype=torch.float) # Rank values
            pred = model(nums).to(device=device, dtype=torch.float) # Predicted Rank values

            error = torch.Tensor.sqrt(torch.Tensor.square(labs-pred).sum()).item()
            print(torch.Tensor.round(labs, decimals=3).tolist(), torch.Tensor.round(pred, decimals=3).tolist(), "---> Error=",round(error,3))
            i+=1
            s_error += error
            nums.detach()
            labs.detach()
    print("avg_error =",round(s_error/i,3))
            
            


# Main code

rank_dat = pd.read_csv(".\\ranking_dataset.csv", dtype=float) # [10000 rows x 20 columns]

rank_dummy, rank_test = train_test_split(rank_dat, test_size=0.2, random_state=42, shuffle=True)

rank_train, rank_val = train_test_split(rank_dat, test_size=0.2, random_state=42, shuffle=True)

# rank_train = (rank_train-np.mean(rank_train,axis=0))/np.std(rank_train,axis=0)
# rank_test = (rank_test-np.mean(rank_train,axis=0))/np.std(rank_train,axis=0)

print("pingx1")
training_data_loader = DataLoader(dataset=RankingDataset(rank_train), batch_size=128, shuffle=True)
validation_data_loader = DataLoader(dataset=RankingDataset(rank_val), batch_size=128, shuffle=True)
print("pingx2")
# Instancing the model(s)
mlp = MLPModel()

loss_history, val_loss_history = MLPtrain(mlp, training_data_loader,validation_data_loader, epochs=500, l_rate=1e-4)

del training_data_loader

#test(mlp, rank_test)
