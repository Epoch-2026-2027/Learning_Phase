import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import MultiHeadAttention
    
# attention seq2seq components

class AttnEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src) # (batch, src_len, embed_dim)
        outputs, hidden = self.rnn(embedded) # outputs: (batch, src_len, hidden_dim * 2)
        outputs = torch.tanh(self.fc(outputs)) # (batch, src_len, hidden_dim)

        if isinstance(hidden, tuple): # LSTM
            h, c = hidden
            hidden = (self._merge(h), self._merge(c))
        else:
            hidden = self._merge(hidden)

        return outputs, hidden

    def _merge(self, h):
        # h: (num_layers * 2, batch, hidden_dim) -> (num_layers, batch, hidden_dim)
        num_layers = h.shape[0] // 2
        h = h.view(num_layers, 2, h.shape[1], self.hidden_dim)
        return torch.tanh(h[:, 0] + h[:, 1])
    

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, cell):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads=1)
        
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden, enc_outputs):
        tgt_token = tgt_token.unsqueeze(1) # (batch, 1)
        embedded = self.embedding(tgt_token) # (batch, 1, embed_dim)
 
        # query is current decoder hidden state
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1) # (batch, 1, hidden_dim)
        else:
            query = hidden[-1].unsqueeze(1) # (batch, 1, hidden_dim)

        context, attn_weights = self.attention(query, enc_outputs, enc_outputs)

        rnn_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden) 
        prediction = self.fc_out(output.squeeze(1)) 

        return prediction, hidden, attn_weights
    
    
class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, buggy, fixed, teacher_forcing_ratio=0.5):
        batch_size = buggy.shape[0]
        fixed_len = fixed.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, fixed_len - 1, vocab_size).to(self.device)
        enc_outputs, hidden = self.encoder(buggy)

        input_token = fixed[:, 0]

        for t in range(1, fixed_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, enc_outputs)
            outputs[:, t - 1] = prediction

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_token = fixed[:, t] if use_teacher else prediction.argmax(dim=-1)

        return outputs

    @torch.no_grad()
    def generate(self, src, max_len=150):
        self.eval()
        if src.dim() == 1:
            src = src.unsqueeze(0)

        enc_outputs, hidden = self.encoder(src)
        input_token = torch.tensor([1], device=self.device)  # <sos>
        generated = []

        for _ in range(max_len):
            prediction, hidden, attn_weights = self.decoder(input_token, hidden, enc_outputs)
            next_token = prediction.argmax(dim=-1)
            if next_token.item() == 2:
                break
            generated.append(next_token.item())
            input_token = next_token

        return generated