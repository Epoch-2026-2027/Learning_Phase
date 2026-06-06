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

device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 

print(torch.cuda.get_device_name(0))


def CharTokenizer(data, max_seq_len, sample_data, purp='src'):
    if purp=='src':
        length = sample_data if sample_data!=None else len(data)
        data_src = []
        for i in range(length):
            seq_len = len(data[i])
            seq_len = seq_len if seq_len<(max_seq_len-1) else max_seq_len-1
            data_src.append(data[i][:seq_len] + chr(3) + chr(1)*(max_seq_len-1-seq_len))
        data_src = np.array([list(s) for s in data_src])
        tokdata_src = torch.tensor(np.vectorize(ord)(data_src.reshape(data_src.shape[0],-1)), dtype=torch.int, device=device)
        return tokdata_src
    if purp=='trg':
        length = sample_data if sample_data!=None else len(data)
        data_infer = []
        data_train = []
        for i in range(length):
            seq_len = len(data[i])
            seq_len = seq_len if seq_len<(max_seq_len-1) else max_seq_len-1
            data_infer.append(chr(2) + data[i][:seq_len] + chr(1)*(max_seq_len-1-seq_len))
            data_train.append(data[i][:seq_len] + chr(3) + chr(1)*(max_seq_len-1-seq_len))
        data_infer = np.array([list(s) for s in data_infer])
        data_train = np.array([list(s) for s in data_train])
        # print(data_infer.shape, data_train.shape)
        tokdata_infer = torch.tensor(np.vectorize(ord)(data_infer.reshape(data_infer.shape[0],-1)), dtype=torch.int, device=device)
        tokdata_train = torch.tensor(np.vectorize(ord)(data_train.reshape(data_train.shape[0],-1)), dtype=torch.int, device=device)
        return (tokdata_infer,tokdata_train)
        

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
        self.buggy, self.fixed_infer, self.fixed_train = tokenizer.tokenize(data=data) #, sample_data=sample_data,

    def __len__(self):
        return torch.max(self.id)+1

    def __getitem__(self, idx):
        return (self.buggy[idx], self.fixed_infer[idx], self.fixed_train[idx])
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dmodel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        posi = torch.arange(max_seq_len).unsqueeze(1).to(device)
        dim = torch.arange(dmodel).unsqueeze(0).to(device)
        self.pos_enc = torch.sin(posi/1000**(2*(dim-dim%2)/dmodel))*(1-dim%2) + torch.cos(posi/1000**(2*(dim-dim%2)/dmodel))*(dim%2) 
        self.max_seq_len = max_seq_len
        self.dmodel = dmodel
        self.key = torch.zeros(self.dmodel).to(device)
    
    def forward(self, input, seq_len=None):
        if seq_len==None:
            seq_len=self.max_seq_len
        out = input + self.pos_enc[:seq_len]
        
        # To prevent PE being added on top of padded tokens, the following implementation for pad-masking
        pad_mask = (torch.abs(input-self.key)>1e-9).all(dim=-1).to(torch.float).unsqueeze(-1)
        out = out * pad_mask

        return out
    


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
    def __init__(self, h, dmodel, dk, dv, masked=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h
        self.masked = masked
        self.Wq = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wk = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wv = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dv)))
        self.Wf = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dv*h, dmodel)))
        self.dropout = nn.Dropout(0.3)

    def forward(self, X):
        # print(X.shape, X.device,'\n\n')
        X = torch.stack(torch.split(X, self.dmodel//self.h, dim=2), dim=1)
        # print(X.shape)
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(X,self.Wk)
        V = torch.matmul(X,self.Wv)
        batchscore = torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(self.dk)
        if self.masked:
            causal_mask = ~torch.flip(torch.triu(torch.ones_like(batchscore), diagonal=0).to(torch.bool), dims=(-1,-2))
            batchscore.masked_fill_(causal_mask, -torch.inf)
            # print(batchscore, batchscore.shape)
            # raise Exception("debug")
        batchscore = torch.softmax(batchscore, dim=-1)
        batchscore = self.dropout(batchscore)
        headattention = torch.matmul(batchscore,V)
        # headattention.shape # batchsize, heads, tokens, dmodel//heads
        attention = torch.flatten(torch.permute(headattention, (0,2,1,3)), start_dim=2, end_dim=3)
        # print(attention.shape)
        out = torch.matmul(attention,self.Wf)
        
        # print("headattention:",headattention[0,:,0],headattention.shape, '\n\n\n', "attention:", attention[0][0],attention.shape, '\n\n\noutput:',out[0][0],out.shape)
        # raise
        return out


class MultiHeadedCrossAttention(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dmodel = dmodel
        self.dk = dk
        self.dv = dv
        self.h = h
        self.Wq = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wk = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dk)))
        self.Wv = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(h, dmodel//h, dv)))
        self.Wf = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(dv*h, dmodel)))
        self.dropout = nn.Dropout(0.3)

    def forward(self, X, Y):
        # print(X.shape, X.device,'\n\n')
        X = torch.stack(torch.split(X, self.dmodel//self.h, dim=2), dim=1)
        Y = torch.stack(torch.split(Y, self.dmodel//self.h, dim=2), dim=1)
        # print(X.shape)
        Q = torch.matmul(X,self.Wq)
        K = torch.matmul(Y,self.Wk)
        V = torch.matmul(Y,self.Wv)
        batchscore = torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(self.dk)
        batchscore = torch.softmax(batchscore, dim=-1)
        batchscore = self.dropout(batchscore)
        headattention = torch.matmul(batchscore,V)
        # headattention.shape # batchsize, heads, tokens, dmodel//heads
        attention = torch.flatten(torch.permute(headattention, (0,2,1,3)), start_dim=2, end_dim=3)
        # print(attention.shape)
        out = torch.matmul(attention,self.Wf)
        
        # print("headattention:",headattention[0,:,0],headattention.shape, '\n\n\n', "attention:", attention[0][0],attention.shape, '\n\n\noutput:',out[0][0],out.shape)
        # raise
        return out



class FFNN(nn.Module):
    def __init__(self, dmodel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp_seq = nn.Sequential(
                                nn.Linear(dmodel, 64),       
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(64, dmodel),
                                nn.Dropout(0.2),
                                nn.ReLU(),   
                            )
    
    def forward(self, input):
        out = self.mlp_seq(input)
        return out

# For the encoder, output must be of shape (Batch_size, seq_len, vector_embed_size)


class EncoderLayer(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multiheadatt = MultiHeadedSelfAttention(h, dmodel, dk, dv)
        self.norm1 = nn.LayerNorm(dmodel)
        self.ffnn = FFNN(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
    
    def forward(self, input):
        x1 = self.multiheadatt(input)
        x1 = self.norm1(x1+input)
        x2 = self.ffnn(x1)
        out = self.norm2(x2+x1)
        return out
    

class DecoderLayer(nn.Module):
    def __init__(self, h, dmodel, dk, dv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maskedmultiheadatt = MultiHeadedSelfAttention(h, dmodel, dk, dv, masked=True)
        self.norm1 = nn.LayerNorm(dmodel)
        self.multiheadcrossatt = MultiHeadedCrossAttention(h, dmodel, dk, dv)
        self.norm2 = nn.LayerNorm(dmodel)
        self.ffnn = FFNN(dmodel)
        self.norm3 = nn.LayerNorm(dmodel)

    def forward(self, decoder_input, encoder_output):
        x1 = self.maskedmultiheadatt(decoder_input)
        # print(x1, x1.shape)
        # raise Exception("test")
        x1 = self.norm1(x1+decoder_input)
        x2 = self.multiheadcrossatt(x1, encoder_output)
        x2 = self.norm2(x2+x1)
        x3 = self.ffnn(x2)
        out = self.norm3(x3+x2)
        return out
    

class Encoder(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, dmodel, pad_id).to(device)
        self.norm = nn.LayerNorm(dmodel)
        self.pos_enc = PositionalEncoding(max_seq_len, dmodel).to(device)
        self.layers = nn.ModuleList([EncoderLayer(h, dmodel, dk, dv).to(device) for _ in range(N)])

    def forward(self, input):
        x = self.embed(input)
        x = self.norm(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, dmodel, pad_id).to(device)
        self.norm = nn.LayerNorm(dmodel)
        self.pos_enc = PositionalEncoding(max_seq_len, dmodel).to(device)
        self.layers = nn.ModuleList([DecoderLayer(h, dmodel, dk, dv).to(device) for _ in range(N)])

    def forward(self, decoder_input, encoder_output, seq_len=None):
        decoder_input = decoder_input.to(device)
        x = self.embed(decoder_input)
        x = self.norm(x)
        x = self.pos_enc(x, seq_len)
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x



# Defining the Transformer based model.
class Transformer(nn.Module):
    def __init__(self, N, h, dmodel, dk, dv, max_seq_len, vocab_size, bos, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos = bos
        self.encoder = Encoder(N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id).to(device)
        self.decoder = Decoder(N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id).to(device)
        self.fc1 = nn.Linear(dmodel, 128).to(device)
        self.fc2 = nn.Linear(128, vocab_size).to(device)

    def forward(self, buggy, fixed=None):
        encoder_output = self.encoder(buggy)
        if fixed is not None:
            #    <=============
            x = self.decoder(fixed, encoder_output)
            x = self.fc1(x)
            out = self.fc2(x)
            return out
        else:
            dummy = torch.ones((encoder_output.shape[0],1), dtype=torch.int)*self.bos
            for _ in range(max_seq_len):
                x = self.decoder(dummy, encoder_output, seq_len=dummy.size(-1))
                x = self.fc1(x)
                x = self.fc2(x)
                dummy = torch.cat((dummy, x.argmax(-1)[:,-1].unsqueeze(-1)), dim=-1)
            return x
            
        




def Transformertrain(model, tr_dataloader, val_dataloader, epochs, l_rate, vocab_size, patience=10, weight_decay=1e-4): 
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
        for buggy, fixed_inf, fixed_tra in tr_dataloader: # Retrieving features for training
            label = fixed_tra.view(-1).long()
            pred = model(buggy, fixed_inf).view(-1, vocab_size)
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
            for buggy, fixed_inf, fixed_tra in val_dataloader:
                label = fixed_tra.view(-1).long()
                pred = model(buggy, fixed_inf).view(-1, vocab_size)
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
            torch.save(model.state_dict(), '.\\Saved Models\\transformer_model.pth')

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f},  Val Loss: {val_loss:.4f}")
        epoch+=1
    return losses, val_losses



def Transformertest(model, tokenizer, test_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    s_matches = 0
    s_editd = 0
    s_bleu = 0
    s_syntax = 0
    i=1
    with torch.no_grad():
        for buggy, fixed_inf, fixed_tra in test_dataloader:
            buggycode = tokenizer.detokenize(buggy.cpu().tolist())
            fixedcode = tokenizer.detokenize(fixed_tra.cpu().tolist())
            # label = fixed_tra.view(-1).long() 
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
train_loader = DataLoader(CodeSet(train_data, tokenizer, sample_data=None), batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(CodeSet(val_data, tokenizer, sample_data=None), batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(CodeSet(test_data, tokenizer, sample_data=None), batch_size=64, shuffle=True, num_workers=0)
print(" Done.\n")

# Initialisation
print("Initialising model...", end='')
model = Transformer(N=1, h=4, dmodel=64, dk=16, dv=16, max_seq_len=max_seq_len, vocab_size=vocab_size, bos=tokenizer.bos, pad_id=tokenizer.pad_id) 
print(" Done.\n")

# Training
# print("Training model... (Press P to abort training)")
# loss_history, val_history = Transformertrain(model, train_loader, val_loader, epochs=100, l_rate=1e-4,
#                                              vocab_size=vocab_size, patience=6, weight_decay=1e-3)


# Loading model with least val_loss
print("Loading model with the best parameters...", end='')
model.load_state_dict(torch.load('.\\Saved Models\\transformer_model.pth'))
print(" Done.\n")

# Testing 
print("Testing...")
Transformertest(model, tokenizer, test_dataloader=test_loader)
print("\nTesting complete!.\n")