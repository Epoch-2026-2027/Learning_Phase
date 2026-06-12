### Project Structure

The project structure was designed to emulate industry-grade codebases.

```
code-repair-seq2seq/
├── data.py          # download, preprocess, tokenize, build vocab, DataLoader
├── baseline.py      # RNN/LSTM/GRU encoder-decoder
├── attention.py     # attention mechanism + attention-based encoder-decoder
├── transformer.py   # transformer components
├── train.py         # training loop, evaluation loop
├── inference.py     # checkpoint loading, metrics, show examples function
└── main.py          # model training + inference, visualization
```