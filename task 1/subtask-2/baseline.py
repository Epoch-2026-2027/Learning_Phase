import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        cls_map = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        rnn_cls = cls_map[model.lower()]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, hidden = self.rnn(embedded)
        return hidden
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        cls_map = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        rnn_cls = cls_map[model.lower()]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_token, hidden):
        target_token = target_token.unsqueeze(1) # since target_token has size (batch, ) (LLM-inspired)
        embedded = self.embedding(target_token)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.output(output.squeeze(1))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, buggy, fixed, teacher_forcing_ratio=0.5): # (LLM-inspired)
        batch_size = buggy.shape[0]
        fixed_len = fixed.shape[1]
        vocab_size = self.decoder.output.out_features # since nn.Linear(hidden_dim, vocab_size), bypasses the need to pass as arg

        outputs = torch.zeros(batch_size, fixed_len - 1, vocab_size).to(self.device) # pre-allocating a tensor to collect predictions at every step
        hidden = self.encoder(buggy) # context vector

        input_token = fixed[:, 0]

        for t in range(1, fixed_len):
            prediction, hidden = self.decoder(input_token, hidden) # self.decoder.forward(...) does the same but other PyTorch features aren't called. 
            outputs[:, t-1] = prediction

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            input_token = fixed[:, t] if use_teacher else prediction.argmax(dim=1) 

        return outputs
    
    def generate(self, buggy, max_len=150): # we generate sequence by sequence because doing batchwise has causes bottleneck (LLM-inspired)
        self.eval()
        with torch.no_grad():
            if buggy.dim() == 1: # (LLM-inspired)
                buggy = buggy.unsqueeze(0) # (1, src_len)
            
            hidden = self.encoder(buggy)
            input_token = torch.tensor([1], device=self.device) # <sos>
            generated = []

            for _ in range(max_len):
                prediction, hidden = self.decoder(input_token, hidden)
                next_token = prediction.argmax(dim=1) # token with highest probability
                if next_token.item() == 2: # <eos>
                    break
                generated.append(next_token.item())
                input_token = next_token

        return generated