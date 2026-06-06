import torch
import pandas as pd
import numpy as np
import re


'''def CharTokenizer(data, max_seq_len, sample_data, purp='src'):
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
        return (tokdata_infer,tokdata_train)'''


class CodeTokenizer():
    def __init__(self, max_seq_len=None):
        if max_seq_len==None:
            self.max_seq_len = 0
            self.findSL = True
        else:
            self.max_seq_len = max_seq_len
            self.findSL = False
        self.vocab = {'\x01':0, '\x02':1, '\x03':2}
    
    def vobularize(self, *datas):
        i=next(reversed(self.vocab.items()))[1]+1
        for set in datas:
            for _, src_data, trg_data in set:
                if self.findSL:
                    self.max_seq_len = max(self.max_seq_len, max(len(src_data), len(trg_data)))
                for word in re.split(r'([. ])', src_data):
                    if word not in self.vocab.keys():
                        self.vocab[word] = i
                        i+=1
                for word in re.split(r'([. ])', trg_data):
                    if word not in self.vocab.keys():
                        self.vocab[word] = i
                        i+=1
        # print(self.vocab)
        # print(self.max_seq_len)
        
    def tokenize(self, data):
        src_tok = []
        trg_tok_inf = []
        trg_tok_tra = []
        for _, src, trg in data:
            src_data = [i for i in re.split(r'([. ])', src) if i!=' ']
            trg_data = [i for i in re.split(r'([. ])', trg) if i!=' ']
            len1 = len(src_data) if len(src_data)<(self.max_seq_len-1) else self.max_seq_len-1
            len2 = len(trg_data) if len(trg_data)<(self.max_seq_len-1) else self.max_seq_len-1
            tokenizing_func = lambda x : dict.get(self.vocab, x)
            src_seq = [(tokenizing_func(i) if tokenizing_func(i)!=None else -1) for i in src_data[:len1] if i!=' '] + [2] + (self.max_seq_len-len1-1)*[0]
            trig_seq = [(tokenizing_func(i) if tokenizing_func(i)!=None else -1) for i in trg_data[:len2] if i!=' ']
            trg_infer_seq= [1] + trig_seq + (self.max_seq_len-len2-1)*[0]
            trg_tra_seq = trig_seq + [2]+ (self.max_seq_len-len2-1)*[0]
            # print(src)
            # print(src_data[:len1])
            # print(src_tok)
            # raise
            src_tok.append(src_seq)
            trg_tok_inf.append(trg_infer_seq)
            trg_tok_tra.append(trg_tra_seq)
        src_tok = torch.tensor(np.array(src_tok), dtype=torch.int, device=torch.device('cpu'))
        trg_tok_inf = torch.tensor(np.array(trg_tok_inf), dtype=torch.int, device=torch.device('cpu'))
        trg_tok_tra = torch.tensor(np.array(trg_tok_tra), dtype=torch.int, device=torch.device('cpu'))
        return src_tok, trg_tok_inf, trg_tok_tra

        

'''
        tokdata_infer = torch.tensor(np.vectorize(ord)(data_infer.reshape(data_infer.shape[0],-1)), dtype=torch.int, device=device)
        tokdata_train = torch.tensor(np.vectorize(ord)(data_train.reshape(data_train.shape[0],-1)), dtype=torch.int, device=device)
        return (tokdata_infer,tokdata_train)'''


train_data = pd.read_parquet('.\\Datasets\\small\\train.parquet').to_numpy()#[:1,:]
wt = CodeTokenizer(max_seq_len=100)
wt.vobularize(train_data)
res = wt.tokenize(train_data)
print(res[0])
print(res[1])