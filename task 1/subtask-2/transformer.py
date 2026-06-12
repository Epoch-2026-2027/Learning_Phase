import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def Attention(Q, K, V, mask=None): # mask is for masked self-attention
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k) # QK^T / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.nn.functional.softmax(scores, dim=-1) # softmax(QK^T / sqrt(d_k))
    return torch.matmul(weights, V), weights # softmax(QK^T / sqrt(d_k)) V
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model) # concatenation of all the heads since d_model = h * d_k
        self.W_K = nn.Linear(d_model, d_model) # this computes all h heads simultaneously and later, we will simply split it
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, X):
        batch_size, seq_len, _ = X.size() # _ is d_model
        return X.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # transpose to bring heads to 2nd axis

    def merge_heads(self, X):
        batch_size, _, seq_len, _ = X.size()
        return X.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_Q(query))
        K = self.split_heads(self.W_K(key))
        V = self.split_heads(self.W_V(value))

        attn_output, attn_weights = Attention(Q, K, V, mask)

        output = self.merge_heads(attn_output)
        output = self.W_O(output)

        return output, attn_weights
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, X):
        return self.network(X)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, X):
        return X + self.pe[:, :X.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, mask=None):
        attn_out, attn_weights = self.attn(X, X, X, mask)
        X = self.norm1(X + attn_out)
        X = self.norm2(X + self.ffn(X))
        return X, attn_weights
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads) # masked self-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # encoder-decoder attention
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, X, enc_output, tgt_mask=None, src_mask=None):
        # masked self-attention
        attn_out, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.norm1(X + attn_out)

        # cross-attention
        attn_out, attn_weights = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.norm2(X + attn_out)

        # feed-forward
        X = self.norm3(X + self.ffn(X))

        return X, attn_weights
    

class TfEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.scale = np.sqrt(d_model)

    def forward(self, src, src_mask=None):
        X = self.embedding(src) * self.scale # (batch, src_len, d_model)
        X = self.pe(X)

        for layer in self.layers:
            X, _ = layer(X, src_mask) # discard attn weights during encoding (LLM-inspired)

        return X 
    

class TfDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = np.sqrt(d_model)

    def forward(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        X = self.embedding(tgt) * self.scale
        X = self.pe(X)

        attn_weights = None

        for layer in self.layers:
            X, attn_weights = layer(
                X,
                enc_output,
                tgt_mask,
                src_mask
            )

        logits = self.fc_out(X)

        return logits, attn_weights
    

class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def make_src_mask(self, src):
        # True where src == PAD, attention ignores these positions
        return (src == 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        tgt_len  = tgt.size(1)
        pad_mask = (tgt == 0).unsqueeze(1).unsqueeze(2) # (batch, 1, 1, tgt_len)
        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=self.device), diagonal=1).bool() # (tgt_len, tgt_len)
        return pad_mask | causal # (batch, 1, tgt_len, tgt_len)

    def forward(self, buggy, fixed, teacher_forcing_ratio=None):
        src_mask = self.make_src_mask(buggy)
        tgt_mask = self.make_tgt_mask(fixed[:, :-1])

        enc_output = self.encoder(buggy, src_mask)
        logits, _ = self.decoder(fixed[:, :-1], enc_output, tgt_mask, src_mask)

        return logits
    

    @torch.no_grad()
    def generate(self, src, max_len=150):
        self.eval()
        if src.dim() == 1:
            src = src.unsqueeze(0)

        src_mask = self.make_src_mask(src)
        enc_output = self.encoder(src, src_mask)
        generated = [1]

        for _ in range(max_len):
            tgt = torch.tensor(generated, device=self.device).unsqueeze(0)
            tgt_mask = self.make_tgt_mask(tgt)
            logits, _ = self.decoder(tgt, enc_output, tgt_mask, src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            if next_token == 2:
                break
            generated.append(next_token)

        return generated[1:]