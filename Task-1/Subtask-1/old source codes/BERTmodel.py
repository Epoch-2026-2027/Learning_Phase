import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enableEmbeds = True
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

def X_y_split(arr, seq_len):
    L = [expand_label(arr[j,seq_len:]) for j in np.arange(arr.shape[0])]
    return arr[:,:seq_len], arr[:,seq_len:]

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
        self.nums = torch.tensor(X_train).to(device=device, dtype=torch.long if (enableEmbeds or not enableNormalisation) else torch.float) # Numerical values
        self.labs = torch.tensor(y_train).to(device=device, dtype=torch.long) # Rank values

    def __len__(self):
        return len(self.nums)
    
    def __getitem__(self, idx):
        return (self.nums[idx], self.labs[idx])


class PositionalEncoding(nn.Module):
    def __init__(self, len_no, dmodel, device,*args, **kwargs):
        super().__init__(*args, **kwargs)
        posi = torch.arange(len_no).unsqueeze(1).to(device)
        dim = torch.arange(dmodel).unsqueeze(0).to(device)
        self.pos_enc = torch.sin(posi/1000**(2*(dim-dim%2)/dmodel))*(1-dim%2) + torch.cos(posi/1000**(2*(dim-dim%2)/dmodel))*(dim%2) 
    
    def forward(self, input):
        return input + self.pos_enc


class SingleHeadedSelfAttention(nn.Module):
    def __init__(self, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.Wq = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wk = nn.Parameter(torch.randn((dmodel, dk)))
        self.Wv = nn.Parameter(torch.randn((dmodel, dv)))
        self.Wf = nn.Parameter(torch.randn((dv, dmodel)))

    def forward(self, X):
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        score = torch.softmax(torch.matmul(Q,K.transpose(-2, -1))/ np.sqrt(self.dk), dim=-1)
        attention = torch.matmul(score,V)
        out = torch.matmul(attention,self.Wf)
        return out
    
class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h
        self.softmax = nn.Softmax(dim=2)
        self.Wq = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wk = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wv = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dv)))
        self.Wf = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dv*h, dmodel)))
        # self.Wq = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dk)))
        # self.Wk = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dk)))
        # self.Wv = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(h, dmodel//h, dv)))
        # self.Wf = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(dv*h, dmodel)))

    def forward(self, X):
        # print(X.shape, X.device,'\n\n')
        X = torch.stack(torch.split(X, self.dmodel//self.h, dim=2), dim=1)
        # print(X.shape)
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        batchscore = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(self.dk), dim=-1)
        headattention = torch.matmul(batchscore,V)
        # headattention.shape # batchsize, heads, tokens, dmodel//heads
        attention = torch.flatten(torch.permute(headattention, (0,2,1,3)), start_dim=-2, end_dim=-1)
        # print(attention.shape)
        out = torch.matmul(attention,self.Wf)
        
        # print("headattention:",headattention[0,:,0],headattention.shape, '\n\n\n', "attention:", attention[0][0],attention.shape, '\n\n\noutput:',out[0][0],out.shape)
        # raise
        return out



class FFNN(nn.Module):
    def __init__(self, dmodel=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        no_input = dmodel if enableEmbeds or dmodel is not None else 1
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_input, 64),       
                                nn.Dropout(0.05),
                                nn.ReLU(),
                                nn.Linear(64, no_input),
                                nn.Dropout(0.05),
                                nn.ReLU(),   
                            )
    
    def forward(self, input):
        out = self.mlp_seq(input)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadatt = MultiHeadedSelfAttention(h, dmodel, dk, dv)
        self.ffnn = FFNN(dmodel)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
    
    def forward(self, input):
        x1 = self.multiheadatt(input)
        # print(x1.shape)
        # print(input.shape)
        # print(x1.device, input.device)
        x1 = self.norm1(x1+input)
        x2 = self.ffnn(x1)
        out = self.norm2(x2+x1)
        return out


# Defining the Transformer-Encoder based model.
class BERTModel(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, no_num=1000, len_no=10, no_labels=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_layers = []
        no_inputs=len_no
        if enableEmbeds:
            self.embed = nn.Embedding(no_num, dmodel).to(device)
            no_inputs*=dmodel
            self.norm = nn.LayerNorm(dmodel).to(device)
        self.pos_enc = PositionalEncoding(len_no, dmodel, device)
        self.encoder_layers = nn.ModuleList([EncoderLayer(h, dmodel, dk, dv).to(device) for _ in range(N)])
        self.flatten = nn.Flatten()
        self.mlp_seq = nn.Sequential(
                                nn.Linear(no_inputs, 128),  
                                nn.Dropout(0.05),     
                                nn.ReLU(),
                                nn.Linear(128, 128),          
                                nn.Dropout(0.05), 
                                nn.ReLU(),
                                nn.Linear(128, 10*no_labels),  
                                nn.Dropout(0.05),
                                nn.Tanh()
                            ).to(device)
    
    def forward(self, input):
        x = self.embed(input)
        x = self.norm(x)
        x = self.pos_enc(x)
        for layer in self.encoder_layers:
            x=layer(x)
        x = self.flatten(x)
        x = self.mlp_seq(x)
        return x
        



def BERTtrain(model, tr_dataloader, val_dataloader, epochs, l_rate, patience=10, weight_decay=1e-4): 
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
            pred = model.forward(nums) 

            # In this model, we have 100 output nodes - 10 at a time representing rank probabilities for
            # a particular position. We use view to convert this (n,100) array to a (n,10,10) array
            # so that the loss is computed, accounting for the dependence of nodes representing same position
            pred = pred.view(-1, 10, 10) # LLM used for this line
            labels = labels.long()       # Long used since we are representing sparse categories
            loss = loss_fn(pred, labels)
            
            total_loss += loss.item()

            loss.backward() # Beginning backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
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


        if losses!=[] and total_loss-losses[-1]>1e-4: # Patience mechanism
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
            torch.save(model.state_dict(), '.\\Saved Models\\bert_model.pth')

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses




def BERTtest(model, test_dataloader):
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
            # print(labs)
            # print(labs.size())
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
X_train, y_train = X_y_split(dat_train, seq_len=10)
X_val, y_val = X_y_split(dat_val, seq_len=10)
X_test, y_test = X_y_split(dat_test, seq_len=10)

# # Normalisations (Z-normalisation)
# X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) 
# y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
# X_train = (X_train-X_mean)/X_std    # Training normalisation
# y_train = (y_train-y_mean)/y_std
# X_val = (X_val-X_mean)/X_std        # Validation normalisation
# y_val = (y_val-y_mean)/y_std
# X_test = (X_test-X_mean)/X_std      # Testing normalisation
# y_test = (y_test-y_mean)/y_std


# Initiating our data loaders
training_data_loader = DataLoader(dataset=RankingDataset(X_train, y_train), batch_size=256, shuffle=True)
validation_data_loader = DataLoader(dataset=RankingDataset(X_val, y_val), batch_size=256, shuffle=True)
testing_data_loader = DataLoader(dataset=RankingDataset(X_test, y_test), batch_size=256, shuffle=True)
print("Test ping! (Loaders initiated)\nInstancing and training model...")

# Instancing the model
bert = BERTModel(N=3, h=2, dmodel=10, dk=5, dv=5, len_no=10, no_labels=10) # Best at N=3, h=2, dmodel=10, dk=5, dv=5, w_d = 1e-4, epoch=300, lr=1e-3

# Training the model
# loss_history, val_loss_history = BERTtrain(bert, training_data_loader,validation_data_loader, epochs=1000, l_rate=1e-3, weight_decay=1e-4)
# print("\nTraining complete! Fetching best model paramaters...")

# Loading best set of model weights
bert.load_state_dict(torch.load('.\\Saved Models\\bert_model_best.pth'))
print("Fetched best model parameters!")

# # Freeing up dataloader from memory
# del training_data_loader

# Testing model
print("\nTesting model!")
# BERTtest(bert, testing_data_loader)
print(bert(torch.tensor([[12, 31, 45, 99, 16, 163, 19, 800, 500, 35],[492,319,4,895,194,573,670,926,58,867]], device=device)).view(-1, 10, 10).argmax(1))
print("Testing done.")
# 4,3,0,8,2,5,6,9,1,7