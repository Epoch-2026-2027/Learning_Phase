import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #


def CharTokenizer(data, max_seq_len):
    for i in range(len(data)):
        seq_len = len(data[i])
        seq_len = seq_len if seq_len<(max_seq_len-1) else max_seq_len-1
        data[i] = data[i][:seq_len] + chr(3) + chr(1)*(max_seq_len-1-seq_len) #chr(2) + at the beginning
    data = np.array(np.array([list(s) for s in data]))
    # print(data)
    tokdata = torch.tensor(np.vectorize(ord)(data.reshape(data.shape[0],-1)), dtype=torch.int, device=device)
    return tokdata


# Dataset class!
class CodeSet(Dataset):
    def __init__(self, data, max_seq_len):
        super().__init__()
        self.id = torch.tensor(data['id'], dtype=torch.int, device=device)
        # Tokenizing np.array of string data before we store them in Dataset
        self.buggy = CharTokenizer(data=data['buggy'].values.tolist(), max_seq_len=max_seq_len)
        self.fixed = CharTokenizer(data=data['fixed'].values.tolist(), max_seq_len=max_seq_len)

    def __len__(self):
        return torch.max(self.id)+1

    def __getitem__(self, idx):
        return (self.buggy[idx], self.fixed[idx])


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(vocab_size)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.enc_rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_size, batch_first=True) # Batchsize, SeqLength, VectorDim

    def forward(self, input):
        x = self.embed(input)
        # print(x.cpu())
        _ , hidden = self.enc_rnn(x)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dec_rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_size, batch_first=True) # Batchsize, SeqLength, VectorDim
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        x = self.embed(input)
        x , hidden_ = self.dec_rnn(x,hidden)
        x = self.fc(x)
        return x, hidden_

class VanillaSeq2Seq(nn.Module):
    def __init__(self, max_seq_len, vocab_size, emb_dim, hidden_size, eos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos = eos
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(vocab_size, emb_dim, hidden_size).to(device)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_size).to(device)

    def forward(self, buggy, fixed=None, tf_ratio=0.5):
        hidden = self.encoder(buggy)

        outputs=[]
        input = torch.zeros(buggy.shape[0], dtype=torch.int).to(device=device)

        for t in range(self.max_seq_len):
            output, hidden = self.decoder(input, hidden)
            outputs.append(output.squeeze(1))
            
            top1 = output.argmax(2).squeeze(-1)
            if fixed is not None: 
                if torch.rand(1).item()<tf_ratio:
                    input = fixed[:,t]
                else:
                    input = top1
            else: # When we dont have a 'fixed' output to train on, like in validation/testing phase
                input = top1
        
        outputs = torch.stack(outputs, dim=0).permute(1,0,2)
        return outputs


model = VanillaSeq2Seq(max_seq_len=120, vocab_size=128, emb_dim=4, hidden_size=32, eos=3)


buggy = CharTokenizer(["public void METHOD_1 ( TYPE_1 VAR_1 ) { VAR_2 . METHOD_2 ( VAR_3 ) ; ( VAR_4 ) ++ ; METHOD_3 ( ) ; } \n",
                          "protected void METHOD_1 ( TYPE_1 VAR_1 ) { TYPE_2 . METHOD_2 ( STRING_1 , STRING_2 ) ; VAR_2 . METHOD_3 ( ) ; } \n"], max_seq_len=120)
fixed = CharTokenizer(["public void METHOD_1 ( TYPE_1 VAR_1 ) { if ( VAR_2 . METHOD_2 ( VAR_3 ) ) { ( VAR_4 ) ++ ; METHOD_3 ( ) ; } } \n",
                          "protected void METHOD_1 ( TYPE_1 VAR_1 ) { VAR_2 . METHOD_3 ( ) ; } \n"], max_seq_len=120)



# print(buggy)

print(fixed, fixed.shape)
labels = torch.nn.functional.one_hot(fixed.to(torch.long), 128)
print(labels, labels.shape)
raise
op = model(buggy,fixed)

print(op,'\n',op.shape) # batchsize, sequence length, vocabulary_probabilities (2, 120, 128)

