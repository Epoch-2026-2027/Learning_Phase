import torch
import torch.nn as nn
import argparse
from data import get_dataloaders
from baseline import Encoder, Decoder, Seq2Seq
from attention import AttnEncoder, AttnDecoder, AttnSeq2Seq
from transformer import TfEncoder, TfDecoder, TransformerSeq2Seq
from train import train
from inference import load_checkpoint, evaluate, show_examples

# args
parser = argparse.ArgumentParser() # took the help of LLM to make sure I didn't miss any arguments + default values

parser.add_argument("--model",      type=str,   default="baseline", choices=["baseline", "attention", "transformer"])
parser.add_argument("--tokenizer",  type=str,   default="whitespace", choices=["whitespace", "char", "bpe"])
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--max_len",    type=int,   default=150)
parser.add_argument("--cell",       type=str,   default="lstm", choices=["rnn", "lstm", "gru"])
parser.add_argument("--embed_dim",  type=int,   default=128)
parser.add_argument("--hidden_dim", type=int,   default=256)
parser.add_argument("--num_layers", type=int,   default=1)
parser.add_argument("--d_model",    type=int,   default=128)
parser.add_argument("--num_heads",  type=int,   default=4)
parser.add_argument("--d_ff",       type=int,   default=256)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--clip",       type=float, default=1.0)
parser.add_argument("--epochs",     type=int,   default=10)

args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
if args.tokenizer == "whitespace":
    tokenizer = str.split
elif args.tokenizer == "char":
    tokenizer = list
elif args.tokenizer == "bpe":
    from data import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=1000)

train_loader, val_loader, test_loader, vocab = get_dataloaders(
    tokenizer = tokenizer,
    batch_size = args.batch_size,
    max_len = args.max_len,
)

# models
if args.model == "baseline":
    EXPERIMENT = f"baseline_{args.cell}_h{args.hidden_dim}_l{args.num_layers}"
    encoder = Encoder(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.cell)
    decoder = Decoder(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.cell)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
elif args.model == "attention":
    EXPERIMENT = f"attention_{args.cell}_h{args.hidden_dim}_l{args.num_layers}"
    encoder = AttnEncoder(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.cell)
    decoder = AttnDecoder(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers, args.cell)
    model = AttnSeq2Seq(encoder, decoder, DEVICE).to(DEVICE)
elif args.model == "transformer":
    EXPERIMENT = f"transformer_dm{args.d_model}_h{args.num_heads}_l{args.num_layers}"
    encoder = TfEncoder(len(vocab), args.d_model, args.num_heads, args.num_layers, args.d_ff)
    decoder = TfDecoder(len(vocab), args.d_model, args.num_heads, args.num_layers, args.d_ff)
    model = TransformerSeq2Seq(encoder, decoder, DEVICE).to(DEVICE)

print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# train
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0) # tells to ignore padded values

train(
    model           = model,
    train_loader    = train_loader,
    val_loader      = val_loader,
    optimizer       = optimizer,
    criterion       = criterion,
    clip            = args.clip,
    epochs          = args.epochs,
    device          = DEVICE,
    experiment_name = EXPERIMENT,
)

# evaluate
model = load_checkpoint( model, EXPERIMENT, DEVICE )
metrics = evaluate(model, test_loader, vocab, DEVICE)
print(metrics)
show_examples(model, test_loader, vocab, DEVICE, n=5)

"""
python main.py --model baseline     --cell lstm --hidden_dim 256 --epochs 10
python main.py --model baseline     --cell gru  --hidden_dim 256 --epochs 10
python main.py --model attention    --cell lstm --hidden_dim 256 --epochs 10
python main.py --model transformer  --d_model 128 --num_heads 4  --epochs 10
"""