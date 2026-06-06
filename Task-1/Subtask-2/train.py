import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import keyboard

from models import CodeTokenizer, CodeSet, CodeSetTransformer, VanillaSeq2Seq, AttendedSeq2Seq, Transformer, device



# Setting a manual seed for reproducability (useful for ablations)
torch.manual_seed(67)

def seed_worker(worker_id): # Manual seeding init function for dataloaders
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)



def train(model, tr_dataloader, val_dataloader, epochs, l_rate, vocab_size, patience=10, weight_decay=1e-4, model_name='model'):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, betas=(0.95,0.99), weight_decay=weight_decay)
    losses = []
    val_losses = []
    best_loss = float('inf') # A large dummy amount, just for initialisation
    pat_count = 0
    epoch=0

    is_transformer = isinstance(model, Transformer)

    while epoch<epochs: # Gradient descent over epochs
        cancel = False
        total_loss=0
        val_loss=0
        model.train()
        i=0
        # with torch.autograd.set_detect_anomaly(True):
        for batch in tr_dataloader: # Retrieving features for training
            if keyboard.is_pressed('alt+c') and epochs>1:   # A training "cancel" function if epochs seem unfavourable
                            cancel=True
                            break
            if is_transformer:
                buggy, fixed_inf, fixed_tra = batch
                label = fixed_tra.view(-1).long()
                pred = model(buggy, fixed_inf).view(-1, vocab_size)
            else:
                buggy, fixed = batch
                label = fixed.long()
                pred = model(buggy, fixed).permute(0,2,1)
            # print(pred, label)
            loss = loss_fn(pred, label)
            total_loss += loss.item()

            loss.backward() # Beginning backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        total_loss/=i+1e-8

        model.eval()
        i=0
        with torch.no_grad():
            for batch in val_dataloader:
                if (keyboard.is_pressed('alt+c') and epochs>1):   # A training "cancel" function if epochs seem unfavourable
                    cancel=True
                    break
                elif cancel==True:
                    break
                if is_transformer:
                    buggy, fixed_inf, fixed_tra = batch
                    label = fixed_tra.view(-1).long()
                    pred = model(buggy, fixed_inf).view(-1, vocab_size)
                else:
                    buggy, fixed = batch
                    label = fixed.long()
                    pred = model(buggy).permute(0,2,1)
                loss = loss_fn(pred, label)
                val_loss+=loss.item()

                i+=1
        val_loss/=i+1e-8

        if cancel:
            break

        if val_losses!=[] and val_loss>=val_losses[-1]: # Patience mechanism
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
            torch.save(model.state_dict(), f'.\\Saved Models\\{model_name}.pth')

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses



#  Main

# Loading our dataset!
print("Loading datasets...", end='')
train_data = pd.read_parquet('.\\Datasets\\small\\train.parquet').to_numpy()
val_data = pd.read_parquet('.\\Datasets\\small\\validation.parquet').to_numpy()
test_data = pd.read_parquet('.\\Datasets\\small\\test.parquet').to_numpy()
print(" Done.\n")

max_seq_len = 150 # Fixed value, seems to fit well for most of the dataset

# Tokenizer set up
print("Setting up tokenizer...", end='')
tokenizer = CodeTokenizer(max_seq_len)
tokenizer.vocabularize(train_data)
vocab_size = len(tokenizer.vocab)
print(" Done.\n")

# Select model here: "vanilla", "attended", "transformer"
selected_model = "transformer"

# Setting up dataloaders
print("Initialising dataloaders...", end='')
if selected_model == "transformer":
    DatasetClass = CodeSetTransformer
    batch_size = 64
else:
    DatasetClass = CodeSet
    batch_size = 256

train_loader = DataLoader(DatasetClass(train_data, tokenizer, sample_data=None), batch_size=batch_size, shuffle=True, num_workers=0,  worker_init_fn=seed_worker)
val_loader = DataLoader(DatasetClass(val_data, tokenizer, sample_data=None), batch_size=batch_size, shuffle=True, num_workers=0,  worker_init_fn=seed_worker)
# test_loader = DataLoader(DatasetClass(test_data, tokenizer, sample_data=None), batch_size=batch_size, shuffle=True, num_workers=0)
print(" Done.\n")

# Initialisation
print("Initialising model...", end='')

if selected_model == "vanilla":
    model = VanillaSeq2Seq(max_seq_len=max_seq_len, vocab_size=vocab_size, emb_dim=8, hidden_size=16, pad_id=tokenizer.pad_id, tf_ratio=0.5)
    model_name = "rnn_model"
elif selected_model == "attended":
    model = AttendedSeq2Seq(max_seq_len=max_seq_len, vocab_size=vocab_size, emb_dim=8, hidden_size=16, pad_id=tokenizer.pad_id, tf_ratio=0.5)
    model_name = "attended_rnn_model"
elif selected_model == "transformer":
    model = Transformer(N=3, h=4, dmodel=64, dk=16, dv=16, max_seq_len=max_seq_len, vocab_size=vocab_size, bos=tokenizer.bos, pad_id=tokenizer.pad_id)
    model_name = "transformer_model"
print(" Done.\n")

# Training
print("Training model... (Press ALT+C to interrupt training)")
loss_history, val_history = train(model, train_loader, val_loader, epochs=100, l_rate=1e-3,
                                  vocab_size=vocab_size, patience=6, weight_decay=1e-2, model_name=model_name)


# Recording training loss / validation loss graph
plt.plot(np.arange(1, len(loss_history)+1),loss_history, color='b', label='training loss')
plt.plot(np.arange(1, len(val_history)+1),val_history, color='r', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss History')
plt.legend()
plt.savefig(f'.\\Graphs and Images\\{model_name}_loss.png')