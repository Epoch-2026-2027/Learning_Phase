import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
# For evaluation metrics
import sacrebleu
import javalang
import Levenshtein

import keyboard
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #

print(torch.cuda.get_device_name(0))


def CharTokenizer(data, max_seq_len):
    for i in range(len(data)):
        seq_len = len(data[i])
        seq_len = seq_len if seq_len<(max_seq_len-1) else max_seq_len-1
        data[i] = data[i][:seq_len] + chr(3) + chr(1)*(max_seq_len-1-seq_len) #chr(2) + at the beginning
    data = np.array(np.array([list(s) for s in data]))
    # print(data)
    tokdata = torch.tensor(np.vectorize(ord)(data.reshape(data.shape[0],-1)), dtype=torch.int, device=device)
    return tokdata

class CodeTokenizer():
    def __init__(self, max_seq_len=None):
        if max_seq_len==None:
            self.max_seq_len = 0
            self.findSL = True
        else:
            self.max_seq_len = max_seq_len
            self.findSL = False
        self.bos = 2
        self.eos = 3
        self.pad_id = 1
        self.unk = 0
        self.del1 = 4
        self.vocab = {'<UNK>':self.unk, '<PAD>':self.pad_id, '<BOS>':self.bos, '<EOS>':self.eos, '.':self.del1}
        self.tokenizing_func = lambda x : dict.get(self.vocab, x)
        self.detokenizing_func = lambda x : list(self.vocab)[x]

    def vocabularize(self, *data):
        i=next(reversed(self.vocab.items()))[1]+1
        for set in data:
            for _, src_data, trg_data in set:
                if self.findSL:
                    self.max_seq_len = max(self.max_seq_len, max(len(src_data), len(trg_data)))
                for word in re.split(r'([. ])', src_data):
                    if word not in self.vocab.keys() and word!=' ':
                        self.vocab[word] = i
                        i+=1
                for word in re.split(r'([. ])', trg_data):
                    if word not in self.vocab.keys() and word!=' ':
                        self.vocab[word] = i
                        i+=1
        
    def tokenize(self, data):
        src_tok = []
        trg_tok_inf = []
        trg_tok_tra = []
        for _, src, trg in data:
            src_data = [i for i in re.split(r'([. ])', src) if i!=' ']
            trg_data = [i for i in re.split(r'([. ])', trg) if i!=' ']
            len1 = len(src_data) if len(src_data)<(self.max_seq_len-1) else self.max_seq_len-1
            len2 = len(trg_data) if len(trg_data)<(self.max_seq_len-1) else self.max_seq_len-1
            
            src_seq = [(self.tokenizing_func(i) if self.tokenizing_func(i)!=None else self.unk) for i in src_data[:len1] if i!=' '] + [self.eos] + (self.max_seq_len-len1-1)*[self.pad_id]
            trig_seq = [(self.tokenizing_func(i) if self.tokenizing_func(i)!=None else self.unk) for i in trg_data[:len2] if i!=' ']
            trg_infer_seq= [self.bos] + trig_seq + (self.max_seq_len-len2-1)*[self.pad_id]
            trg_tra_seq = trig_seq + [self.eos]+ (self.max_seq_len-len2-1)*[self.pad_id]
            src_tok.append(src_seq)
            trg_tok_inf.append(trg_infer_seq)
            trg_tok_tra.append(trg_tra_seq)

        src_tok = torch.tensor(np.array(src_tok), dtype=torch.int, device=device)
        trg_tok_inf = torch.tensor(np.array(trg_tok_inf), dtype=torch.int, device=device)
        trg_tok_tra = torch.tensor(np.array(trg_tok_tra), dtype=torch.int, device=device)
        return src_tok, trg_tok_inf, trg_tok_tra
    
    def detokenize(self, tok_data):
        ac_data=[]
        for tok_seq in tok_data:
            seq = ""
            for t in range(len(tok_seq)):
                if tok_seq[min(t+1, len(tok_seq)-1)] in (self.eos, self.pad_id, self.del1) or tok_seq[t]==self.del1:
                    seq += self.detokenizing_func(tok_seq[t])
                else:
                    seq += self.detokenizing_func(tok_seq[t]) + ' '
            seq = re.sub('<EOS>|<PAD>', '', seq).rstrip()
            ac_data.append(seq)
        return ac_data



# Dataset class!
class CodeSet(Dataset):
    def __init__(self, data, tokenizer, sample_data=None):
        super().__init__()
        # length = sample_data if sample_data!=None else len(data)
        self.id = torch.tensor(data[:,0].astype(int), dtype=torch.int, device=device) #[:length]
        # Tokenizing np.array of string data before we store them in Dataset
        self.buggy, _, self.fixed = tokenizer.tokenize(data=data) #, sample_data=sample_data,

    def __len__(self):
        return torch.max(self.id)+1

    def __getitem__(self, idx):
        return (self.buggy[idx], self.fixed[idx])


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim, pad_id)
        self.enc_rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True) # Batchsize, SeqLength, VectorDim

    def forward(self, input):
        x = self.embed(input)
        # print(x.cpu())
        _ , hidden = self.enc_rnn(x)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim, pad_id)
        self.dec_rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True) # Batchsize, SeqLength, VectorDim
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        x = self.embed(input)
        x , hidden_ = self.dec_rnn(x,hidden)
        x = self.fc(x)
        return x, hidden_

class VanillaSeq2Seq(nn.Module):
    def __init__(self, max_seq_len, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size=vocab_size
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)

    def forward(self, buggy, fixed=None, tf_ratio=0.5):
        hidden = self.encoder(buggy)

        outputs=[]
        input = torch.zeros(buggy.shape[0], dtype=torch.int).to(device=device)

        for t in range(self.max_seq_len):
            output, hidden = self.decoder(input, hidden)
            outputs.append(output.squeeze(1))
            
            top1 = output.argmax(2).squeeze(-1)
            if fixed is not None and torch.rand(1).item()<tf_ratio:
                input = fixed[:,t]
            else:
                input = top1
        
        outputs = torch.stack(outputs, dim=0).permute(1,0,2)
        return outputs


def RNNtrain(model, tr_dataloader, val_dataloader, epochs, l_rate, vocab_size, patience=10, weight_decay=1e-4): 
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
        if keyboard.is_pressed('p') and epoch!=0: # Implementation to abort training
            break
        total_loss = 0
        val_loss = 0

        model.train()
        i=0
        # with torch.autograd.set_detect_anomaly(True):
        for buggy, fixed in tr_dataloader: # Retrieving features for training
            label = fixed.long()
            # print(label.dtype)
            # raise
            pred = model(buggy, fixed).permute(0,2,1)#.view(-1, vocab_size)
            # print(pred.shape)
            # raise
            # print(pred, label)
            loss = loss_fn(pred, label)
            total_loss += loss.item()

            loss.backward() # Beginning backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            optimizer.step()
            optimizer.zero_grad()
            i+=1
        total_loss/=i

        model.eval()
        i=0
        with torch.no_grad():
            for buggy, fixed in val_dataloader:
                label = fixed.long()
                pred = model(buggy).permute(0,2,1)
                loss = loss_fn(pred, label)
                val_loss+=loss.item()

                i+=1
        val_loss/=i


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
            torch.save(model.state_dict(), '.\\Saved Models\\rnn_model.pth')

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses



def RNNtest(model, tokenizer, test_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    s_matches = 0
    s_editd = 0
    s_bleu = 0
    s_syntax = 0
    i=1
    with torch.no_grad():
        for buggy, fixed in test_dataloader:
            buggycode = tokenizer.detokenize(buggy.cpu().tolist())
            fixedcode = tokenizer.detokenize(fixed.cpu().tolist())
            # label = fixed.view(-1).long() 
            pred = model(buggy)
            # print(pred.shape)
            predcode = tokenizer.detokenize(pred.argmax(-1).cpu().tolist())
            
            for bug,fix,pre in zip(buggycode,fixedcode,predcode):
                match_score = sum([1 if i==j else 0 for i,j in zip(fix,pre)])
                edit_distance = Levenshtein.distance(fix,pre)
                bleu = sacrebleu.sentence_bleu(pre, [fix]).score
                try:
                    _ = javalang.parse.parse(pre)
                    syntax_check = "Valid"
                    s_syntax+=1
                except:
                    syntax_check = "Invalid"

                if i%100==0: # or syntax_check=="Valid"
                    print(f"{i}. " +f"\t[To Debug]>\t{bug}\n\t[Model]>\t{pre}\n\t[Actual]>\t{fix}\n")
                    print(f"----> Match_score= {match_score}\n----> Edit_distance= {edit_distance}\n----> BLEU= {round(bleu,3)}\n----> Syntax_check= {syntax_check}\n\n")
                
                s_matches+=match_score
                s_editd+=edit_distance
                s_bleu+=bleu
                i+=1

            
            
    print("avg_absolute_matches =",round(s_matches/i,3))
    print("avg_edit_distance =",round(s_editd/i,3))
    print("avg_bleu_score =",round(s_bleu/i,3))
    print("fraction_valid_syntax =",round(s_syntax/i,3))

# Loading our dataset!
print("Loading datasets...", end='')
train_data = pd.read_parquet('.\\Datasets\\small\\train.parquet').to_numpy()
val_data = pd.read_parquet('.\\Datasets\\small\\validation.parquet').to_numpy()
test_data = pd.read_parquet('.\\Datasets\\small\\test.parquet').to_numpy()
print(" Done.\n")

max_seq_len = 150

# Tokenizer set up
print("Setting up tokenizer...", end='')
tokenizer = CodeTokenizer(max_seq_len)
tokenizer.vocabularize(train_data)
vocab_size = len(tokenizer.vocab)
print(" Done.\n")

# Setting up dataloaders
print("Initialising dataloaders...", end='')
train_loader = DataLoader(CodeSet(train_data, tokenizer, sample_data=None), batch_size=512, shuffle=True, num_workers=0)
val_loader = DataLoader(CodeSet(val_data, tokenizer, sample_data=None), batch_size=512, shuffle=True, num_workers=0)
test_loader = DataLoader(CodeSet(test_data, tokenizer, sample_data=None), batch_size=512, shuffle=True, num_workers=0)
print(" Done.\n")

# Initialisation
print("Initialising model...", end='')
model = VanillaSeq2Seq(max_seq_len=max_seq_len, vocab_size=vocab_size, emb_dim=8, hidden_size=16, pad_id=tokenizer.pad_id) 
print(" Done.\n")

# Training
print("Training model... (Press P to abort training)")
loss_history, val_history = RNNtrain(model, train_loader, val_loader, epochs=100, l_rate=1e-4,
                                             vocab_size=vocab_size, patience=6, weight_decay=1e-3)


# Loading model with least val_loss
print("Loading model with the best parameters...", end='')
model.load_state_dict(torch.load('.\\Saved Models\\rnn_model.pth'))
print(" Done.\n")

# Testing 
print("Testing...")
RNNtest(model, tokenizer, test_dataloader=test_loader)
print("\nTesting complete!.\n")