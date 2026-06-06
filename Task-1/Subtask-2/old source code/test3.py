import torch

embed_dim=3

input = torch.randn((2,4,3))
input[:,2,:] = 0.0

print(input)

term = torch.arange(1,13).view(4,3)

out = input + term
print(out)
# To prevent PE being added on top of padded tokens, the following implementation for pad-masking
key = torch.zeros(embed_dim)
pad_mask = (torch.abs(input-key)>1e-9).all(dim=-1).to(torch.float).unsqueeze(-1) # (batch_size,seq_len,1).transpose(-1,-2) X batch_size,seq_len,emb_dim   ---> batch_size,seq_len,emb_dim 
print(pad_mask)
out = out * pad_mask
print(out)
