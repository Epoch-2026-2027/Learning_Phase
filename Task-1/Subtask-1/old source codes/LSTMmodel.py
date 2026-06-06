import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enableEmbeds = False
enableNormalisation = False


def expand_label(lab):
    n = len(lab)
    l=[]
    for i in range(n):
        temp = np.zeros((n,1))
        temp[int(lab[i])] = 1
        l.extend(temp)
    exp_labs = np.array(l, dtype=float).flatten()
    return exp_labs

def X_y_split(arr):
    L = [expand_label(arr[j,10:]) for j in np.arange(arr.shape[0])]
    return arr[:,:10], arr[:,10:]

def contract_label(exp_lab):
    n = len(exp_lab)//10
    l=[]
    for i in range(n):
        c=0
        for j in range(i*10,i*10+10):
            if exp_lab[j]!=0:
                l.append(j-i*10)
                c=1
                break
        if not c:
            l.append(-1)
    return np.array(l)



# Creating the Dataset class
class RankingDataset(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nums = torch.tensor(X_train).to(device=device, dtype=torch.float if (enableEmbeds or not enableNormalisation) else torch.long) # Numerical values
        self.labs = torch.tensor(y_train).to(device=device, dtype=torch.float) # Rank values

    def __len__(self):
        return len(self.nums)
    
    def __getitem__(self, idx):
        # print(self.nums[idx], self.labs[idx])
        # raise Exception
        return (self.nums[idx], self.labs[idx])


# Defining the LSTM based model.
class LSTMModel(nn.Module):
    def __init__(self, no_num=1000, len_no=10, no_label=10, emb_dim=1, num_layers=1, bidirectional=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        no_input = len_no*emb_dim if enableEmbeds else len_no
        if enableEmbeds:
            self.embed = nn.Embedding(no_num, emb_dim) #EmbedAndConcat(no_num, 16)
            self.flat = nn.Flatten()
        self.lstm_seq = nn.LSTM(input_size=no_input, hidden_size=128, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(128*2 if bidirectional else 128, 10*no_label)
    
    def forward(self, nums): 
        if enableEmbeds:
            x = self.embed(nums)
            x = self.flat(x)
            #raise Exception
            x = self.lstm_seq(x)[0]
            x =  self.fc(x)
            return x
        else:
            x = self.lstm_seq(nums)[0]
            x =  self.fc(x)
            return x


def LSTMtrain(model, tr_dataloader, val_dataloader, epochs, l_rate, patience=10, weight_decay=1e-4): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Moving to cuda accelerator if possible
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=l_rate) # Stochastic Gradient Descent
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=weight_decay) # Adaptive Moment Estimation
    losses = []
    val_losses = []
    best_loss = 1000000 # A large dummy amount, just for initialisation
    pat_count = 0
    epoch=0
    while epoch<epochs: # Gradient descent over epochs
        total_loss = 0
        val_loss = 0

        model.train()
        i=0
        for nums, labels in tr_dataloader: # Retrieving features for training
            # print(nums, labels)
            # print(nums.shape, labels.shape)
            pred = model.forward(nums) 

            # In this model, we have 100 output nodes - 10 at a time representing rank probabilities for
            # a particular position. We use view to convert this (n,100) array to a (n,10,10) array
            # so that the loss is computed, accounting for the dependence of nodes representing same position
            pred = pred.view(-1, 10, 10) # LLM used for this line
            labels = labels.long()       # Long used since we are representing sparse categories
            loss = loss_fn(pred, labels)
            
            total_loss += loss.item()

            loss.backward() # Beginning backprop
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        total_loss/=i

        model.eval()
        i=0
        for nums, labels in val_dataloader:
            pred = model.forward(nums) 
            pred = pred.view(-1, 10, 10)
            labels = labels.long()   
            loss = loss_fn(pred, labels)
            
            val_loss += loss.item()

            optimizer.zero_grad()
            i+=1
        val_loss/=i


        if losses!=[] and total_loss>losses[-1]: # Patience mechanism
            if pat_count==patience:
                break
            else:
                pat_count+=1
        else:
            pat_count=0

        losses.append(total_loss)
        val_losses.append(val_loss)

        if val_loss<best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), '.\\Saved Models\\lstm_model.pth')

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses




def LSTMtest(model, test_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    s_error = 0
    s_matches = 0
    i=0
    j=0
    with torch.no_grad():
        for nums, labs in test_dataloader:
            pred = model.forward(nums)
            pred = pred.view(-1, 10, 10)
            labs = labs.long()   
            error = loss_fn(pred, labs).item()

            pred_ranks = torch.argmax(pred, dim=1)
            print(labs)
            print(labs.size())
            for lab, pre in zip(labs, pred_ranks):
                sample_true = lab.cpu().tolist()
                sample_pred = pre.cpu().tolist()

                no_matches = sum(1 for p, t in zip(sample_pred, sample_true) if p == t)
                i+=1
                j+=1
                s_error += error
                s_matches += no_matches
                if j%50==0: 
                    print(f"{i}. Predic-->\t",sample_pred, "\nActual-->\t",sample_true, "---> Error=",round(s_error/i,3), "---> No. matches=",no_matches,'\n')
            
            # # Detaching tensors from GPU memory
            # nums.detach()
            # labs.detach()
            # pred.detach()
    print("avg_error =",round(s_error/i,3))
    print("avg_no_matches =",round(s_matches/i,3))    

            


# Main code

#Loading Data
dat = pd.read_csv(".\\ranking_dataset.csv", dtype=int).to_numpy() # [10000 rows x 20 columns]

# Train-Validation-Test split (80-10-10)
dat_dummy, dat_test = train_test_split(dat, test_size=0.1, random_state=42, shuffle=True)
dat_train, dat_val = train_test_split(dat_dummy, test_size=0.1/0.9, random_state=42, shuffle=True)

# Number - Rank splits
X_train, y_train = X_y_split(dat_train)
X_val, y_val = X_y_split(dat_val)
X_test, y_test = X_y_split(dat_test)

# # Normalisations (Z-normalisation)
if enableNormalisation:
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) 
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    X_train = (X_train-X_mean)/X_std    # Training normalisation
    y_train = (y_train-y_mean)/y_std
    X_val = (X_val-X_mean)/X_std        # Validation normalisation
    y_val = (y_val-y_mean)/y_std
    X_test = (X_test-X_mean)/X_std      # Testing normalisation
    y_test = (y_test-y_mean)/y_std


# Initiating our data loaders
training_data_loader = DataLoader(dataset=RankingDataset(X_train, y_train), batch_size=256, shuffle=True)
validation_data_loader = DataLoader(dataset=RankingDataset(X_val, y_val), batch_size=256, shuffle=True)
testing_data_loader = DataLoader(dataset=RankingDataset(X_test, y_test), batch_size=256, shuffle=True)
print("Test ping! (Loaders initiated)\nInstancing and training model...")

# Instancing the model
mlp = LSTMModel(emb_dim=3, num_layers=2, bidirectional=True)

# Training the model
loss_history, val_loss_history = LSTMtrain(mlp, training_data_loader,validation_data_loader, epochs=50, l_rate=3e-5, weight_decay=1e-3)
print("\nTraining complete! Fetching best model paramaters...")

# Loading best set of model weights
mlp.load_state_dict(torch.load('.\\Saved Models\\lstm_model.pth'))
print("Fetched best model parameters!")

# Freeing up dataloader from memory
del training_data_loader

# Testing model
print("\nTesting model!")
LSTMtest(mlp, testing_data_loader)
print("Testing done.")
