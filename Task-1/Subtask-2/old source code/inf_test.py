import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import sacrebleu
import javalang
import Levenshtein
from models import CodeTokenizer, CodeSet, CodeSetTransformer, VanillaSeq2Seq, AttendedSeq2Seq, Transformer, device


def test(model, tokenizer, test_dataloader):
    model.eval()
    s_matches = 0
    s_editd = 0
    s_bleu = 0
    s_syntax = 0
    i=1

    is_transformer = isinstance(model, Transformer)

    with torch.no_grad():
        for batch in test_dataloader:
            if is_transformer:
                buggy, fixed_inf, fixed_tra = batch
                buggycode = tokenizer.detokenize(buggy.cpu().tolist())
                fixedcode = tokenizer.detokenize(fixed_tra.cpu().tolist())
                pred = model(buggy)
                # print(pred.shape)
                predcode = tokenizer.detokenize(pred.argmax(-1).cpu().tolist())
            else:
                buggy, fixed = batch
                buggycode = tokenizer.detokenize(buggy.cpu().tolist())
                fixedcode = tokenizer.detokenize(fixed.cpu().tolist())
                pred = model(buggy)
                # print(pred.shape)
                predcode = tokenizer.detokenize(pred.argmax(-1).cpu().tolist())
            
            for bug,fix,pre in zip(buggycode,fixedcode,predcode):
                match_score = sum([1 if i==j else 0 for i,j in zip(fix,pre)])
                edit_distance = Levenshtein.distance(fix,pre)
                bleu = sacrebleu.sentence_bleu(pre, [fix]).score
                try:
                    _ = javalang.parse.parse(pre)
                    syntax_check = "Valid"
                    s_syntax+=1
                except:
                    syntax_check = "Invalid"

                if i%100==0: # or syntax_check=="Valid"
                    print(f"{i}. " +f"\t[To Debug]>\t{bug}\n\t[Model]>\t{pre}\n\t[Actual]>\t{fix}\n")
                    print(f"----> Match_score= {match_score}\n----> Edit_distance= {edit_distance}\n----> BLEU= {round(bleu,3)}\n----> Syntax_check= {syntax_check}\n\n")
                
                s_matches+=match_score
                s_editd+=edit_distance
                s_bleu+=bleu
                i+=1

    print("avg_absolute_matches =",round(s_matches/i,3))
    print("avg_edit_distance =",round(s_editd/i,3))
    print("avg_bleu_score =",round(s_bleu/i,3))
    print("fraction_valid_syntax =",round(s_syntax/i,3))



#  Main

# Loading our dataset! using only small code set for easier and faster computing
print("Loading datasets...", end='')
train_data = pd.read_parquet('.\\Datasets\\small\\train.parquet').to_numpy()
val_data = pd.read_parquet('.\\Datasets\\small\\validation.parquet').to_numpy()
test_data = pd.read_parquet('.\\Datasets\\small\\test.parquet').to_numpy()
print(" Done.\n")

max_seq_len = 150

# Tokenizer set up
print("Setting up tokenizer...", end='')
tokenizer = CodeTokenizer(max_seq_len)
tokenizer.vocabularize(train_data)
vocab_size = len(tokenizer.vocab)
print(" Done.\n")

# Select model here: "vanilla", "attended", "transformer"
selected_model = "transformer"

# Setting up dataloaders
print("Initialising dataloaders...", end='')
if selected_model == "transformer":
    DatasetClass = CodeSetTransformer
    batch_size = 64
else:
    DatasetClass = CodeSet
    batch_size = 256

test_loader = DataLoader(DatasetClass(test_data, tokenizer, sample_data=None), batch_size=batch_size, shuffle=True, num_workers=0)
print(" Done.\n")

# Initialisation
print("Initialising model...", end='')
if selected_model == "vanilla":
    model = VanillaSeq2Seq(max_seq_len=max_seq_len, vocab_size=vocab_size, emb_dim=8, hidden_size=16, pad_id=tokenizer.pad_id, tf_ratio=0.5)
    model_name = "rnn_model"
elif selected_model == "attended":
    model = AttendedSeq2Seq(max_seq_len=max_seq_len, vocab_size=vocab_size, emb_dim=8, hidden_size=16, pad_id=tokenizer.pad_id, tf_ratio=0.5)
    model_name = "attended_rnn_model"
elif selected_model == "transformer":
    model = Transformer(N=1, h=4, dmodel=64, dk=16, dv=16, max_seq_len=max_seq_len, vocab_size=vocab_size, bos=tokenizer.bos, pad_id=tokenizer.pad_id)
    model_name = "transformer_model"
print(" Done.\n")

# Loading model with least val_loss
print("Loading model with the best parameters...", end='')
model.load_state_dict(torch.load(f'.\\Saved Models\\{model_name}.pth'))
print(" Done.\n")

# Testing
print("Testing...")
test(model, tokenizer, test_dataloader=test_loader)
print("\nTesting complete!.\n")
