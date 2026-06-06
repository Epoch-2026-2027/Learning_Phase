import torch
from torch import nn
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  Tokenizers
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

from torch.utils.data import Dataset

# Dataset class! (Transformer version — returns buggy, fixed_infer, fixed_train)
class CodeSetTransformer(Dataset):
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

# Dataset class! (RNN version — returns buggy, fixed)
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


# ─────────────────────────────────────────────
#  Vanilla Seq2Seq (GRU)
# ─────────────────────────────────────────────

class VanillaEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim, pad_id)
        self.enc_rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, batch_first=True) # Batchsize, SeqLength, VectorDim

    def forward(self, input):
        x = self.embed(input)
        # print(x.cpu())
        _ , hidden = self.enc_rnn(x)
        return hidden


class VanillaDecoder(nn.Module):
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
        self.encoder = VanillaEncoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)
        self.decoder = VanillaDecoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)

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


# ─────────────────────────────────────────────
#  Attended Seq2Seq (biLSTM + Attention)
# ─────────────────────────────────────────────

class AttendedEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.enc_rnn = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True) # Batchsize, SeqLength, VectorDim
        self.drop = nn.Dropout(0.3)

    def forward(self, input):
        # input = input.unsqueeze(1)
        x = self.embed(input)
        # print(x.shape)
        # print(x.cpu())
        enc_hidden , _ = self.enc_rnn(x)
        out = self.drop(enc_hidden)
        return out


class AttendedDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, pad_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dec_rnn = nn.LSTM(input_size=emb_dim+2*hidden_size, hidden_size=2*hidden_size, num_layers=1, batch_first=True) # Batchsize, SeqLength, VectorDim
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2*hidden_size, vocab_size)

    def forward(self, input, hidden, cell, context_vector):
        input = input.unsqueeze(1)
        x = self.embed(input)
        x = torch.concatenate([x,context_vector],dim=-1)
        x , (hidden, cell) = self.dec_rnn(x,(hidden, cell))
        x = self.drop(x)
        x = self.fc(x)
        return x, (hidden, cell)

class AttendedSeq2Seq(nn.Module):
    def __init__(self, max_seq_len, vocab_size, emb_dim, hidden_size, pad_id, tf_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size=vocab_size
        self.max_seq_len = max_seq_len
        self.tf_ratio = tf_ratio
        self.encoder = AttendedEncoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)
        self.decoder = AttendedDecoder(vocab_size, emb_dim, hidden_size, pad_id).to(device)

    def forward(self, buggy, fixed=None):
        enc_hiddens = self.encoder(buggy)
        
        outputs=[]
        input = torch.zeros(buggy.shape[0], dtype=torch.int).to(device=device)

        hidden = enc_hiddens[:,-1,:].unsqueeze(-2)
        cell = torch.zeros_like(hidden)

        for t in range(self.max_seq_len):
            # Attention mechanism
            context_scores = torch.softmax(torch.matmul(hidden, enc_hiddens.transpose(-1,-2)),dim=-1)
            context_vector = torch.matmul(context_scores, enc_hiddens)
            # (DEBUG)print(hidden.shape, enc_hiddens.shape)
            # (DEBUG)print(t, context_scores.shape, '=', hidden.shape, 'X', enc_hiddens.transpose(-1,-2).shape)
            # (DEBUG)raise
            # (DEBUG)print(context_vector.shape, hidden.shape)
            # (DEBUG)raise Exception("testing testing ahh")
            
            # Forward pass
            hidden = hidden.permute(1,0,2)
            cell = cell.permute(1,0,2)
            output, (hidden, cell) = self.decoder(input, hidden, cell, context_vector)
            hidden = hidden.permute(1,0,2)
            cell = cell.permute(1,0,2)
            outputs.append(output.squeeze(1))
            
            top1 = output.argmax(2).squeeze(-1)
            if torch.rand(1).item()<self.tf_ratio and fixed is not None:
                input = fixed[:,t]
            else: # When we dont have a 'fixed' output to train on, like in validation/testing phase, or teacher forcing doesnt occur
                input = top1
            
        outputs = torch.stack(outputs, dim=0).permute(1,0,2)
        return outputs


# ─────────────────────────────────────────────
#  Transformer
# ─────────────────────────────────────────────

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
    

class TransformerEncoder(nn.Module):
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
    
class TransformerDecoder(nn.Module):
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
        self.max_seq_len = max_seq_len
        self.encoder = TransformerEncoder(N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id).to(device)
        self.decoder = TransformerDecoder(N, h, dmodel, dk, dv, max_seq_len, vocab_size, pad_id).to(device)
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
            for _ in range(self.max_seq_len):
                x = self.decoder(dummy, encoder_output, seq_len=dummy.size(-1))
                x = self.fc1(x)
                x = self.fc2(x)
                dummy = torch.cat((dummy, x.argmax(-1)[:,-1].unsqueeze(-1)), dim=-1)
            return x
