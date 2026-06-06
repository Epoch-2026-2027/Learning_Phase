import torch
import numpy as np
# import matplotlib.pyplot as plt

h=3
dmodel=12
dk=3
dv=3
len_no=4

X = torch.randn(96)
# print(X, X.shape, '\n')
X = X.view(2,4,dmodel) # sequence length 4, batch length 2
print(X, X.shape, '\n')


# pos_enc = torch.stack((prod1,prod2),dim=2).view(len_no, dmodel)
posi = torch.arange(len_no).unsqueeze(1)
dim = torch.arange(dmodel).unsqueeze(0)
pos_enc = torch.sin(posi/1000**(2*(dim-dim%2)/dmodel))*(1-dim%2) + torch.cos(posi/1000**(2*(dim-dim%2)/dmodel))*(dim%2) 
print(pos_enc, pos_enc.shape)

# plt.imshow(pos_enc, cmap='plasma', interpolation='nearest')
# plt.show()

encoded_X = X + pos_enc
print(encoded_X, encoded_X.shape)
#X = X.view(h, -1, dmodel//h)#.transpose(-1,-2)
# X = torch.tensor(np.array([X[:,j:j+dmodel//h] for j in range(0, dmodel, dmodel//h)]), dtype=torch.float)
# X = torch.stack(torch.hsplit(X, h))
# print(X, X.shape, '\n') # 3,8,4


# Wk = torch.randn(dmodel*dk).view(h, dmodel//h, dk)
# # print(W, W.shape, '\n') # 3,40,3
# Wq = torch.randn(dmodel*dk).view(h, dmodel//h, dk)
# Wv = torch.randn(dmodel*dv).view(h, dmodel//h, dv)
# print(Wq.shape, Wk.shape, Wv.shape,'\n')

# Q = torch.matmul(X,Wq)
# K = torch.matmul(X,Wk)
# V = torch.matmul(X,Wv)

# print(Q.shape, K.shape, V.shape)

# batchscore = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/np.sqrt(dk), dim=2)
# print(batchscore, batchscore.shape, '\n')

# headattention = torch.matmul(batchscore,V)
# print(headattention, headattention.shape,'\n') # (h, n, dv)
# attention = torch.concatenate(list(headattention),dim=1)
# print(attention, attention.shape,'\n') # (n, dv*h)

# W = torch.randn(dmodel*dv*h).view(dv*h, dmodel)
# print(W.shape,'\n')

# res = torch.matmul(attention,W)
# print(res, res.shape)
