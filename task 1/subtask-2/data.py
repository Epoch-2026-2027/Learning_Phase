import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

"""
This file takes care of downloading the raw data, preprocessing that raw data, tokenizing the preprocessed data,
building vocab, building a DataLoader pipeline that pads different length sequences per batch
"""

class Vocab:
    def __init__(self):
        self.tokentoidx = {}
        self.idxtotoken = {}

        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        for token in special_tokens:
            idx = len(self.tokentoidx)
            self.tokentoidx[token] = idx
            self.idxtotoken[idx] = token

    def build(self, token_sequences):
        for tokens in token_sequences:
            for token in tokens:
                if token not in self.tokentoidx:
                    idx = len(self.tokentoidx)
                    self.tokentoidx[token] = idx
                    self.idxtotoken[idx] = token

    def encode(self, tokens):
        unk = self.tokentoidx["<unk>"]
        return [self.tokentoidx.get(token, unk) for token in tokens]

    def decode(self, indices):
        return [self.idxtotoken.get(idx, "<unk>") for idx in indices]

    def __len__(self):
        return len(self.tokentoidx)
    

class CharTokenizer:
    def __init__(self):
        self.chartoidx = {}
        self.idxtochar = {}

    def build(self, texts): # builds its vocab on the given text
        chars = sorted(set(ch for text in texts for ch in text))
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"] # padding token for padding a shorter sentence, starting & ending of sentence token, unkown token
        vocab = specials + chars # better to keep specials first, since it will have the indices 0, 1, 2, 3 always

        self.chartoidx = {ch: i for i, ch in enumerate(vocab)}
        self.idxtochar = {i: ch for ch, i in self.chartoidx.items()}

    def encode(self, text): # text to indices
        unk = self.chartoidx["<unk>"]
        return [self.chartoidx.get(ch, unk) for ch in text] # .get(ch, unk) says if ch is found, return its id. if not, return unk (LLM-inspired)
    
    def decode(self, indices): # indices to text
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        return "".join(self.idxtochar[idx] for idx in indices
                       if self.idxtochar.get(idx) not in specials) # simply ignoring special tokens since we can't output them


class WhitespaceTokenizer():
    def __init__(self):
        self.wordtoidx = {}
        self.idxtoword = {}

    def build(self, texts):
        unique_words = set( # set to remove duplicates
            word for text in texts
            for word in text.split()
        )
        words = sorted(list(unique_words))

        specials = ["<pad>", "<sos>", "<eos>", "<unk>"] # 
        vocab = specials + words

        self.wordtoidx = {word: i for i, word in enumerate(vocab)}
        self.idxtoword = {i: word for word, i in self.wordtoidx.items()}

    def encode(self, text):
        unk = self.wordtoidx["<unk>"]
        return [self.wordtoidx.get(word, unk) for word in text.split()]
    
    def decode(self, indices):
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        return " ".join(self.idxtoword[idx] for idx in indices
                       if self.idxtoword.get(idx) not in specials) 


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.tokentoidx = {}
        self.idxtotoken = {}

    @staticmethod # since this is only a utility function, each inheritance does not require it
    def get_stats(indices):
        counts = {}
        for pair in zip(indices, indices[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(indices, pair, idx):
        new_indices = []
        i = 0
        while i < len(indices):
            if i < len(indices) - 1 and indices[i] == pair[0] and indices[i+1] == pair[1]:
                new_indices.append(idx)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices

    def build(self, texts):
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.tokentoidx = {tok: i for i, tok in enumerate(specials)}
        self.idxtotoken = {i: tok for i, tok in enumerate(specials)}

        text = " ".join(texts)
        chars = sorted(set(text)) # starting with byte-level vocabulary: every unique character gets an id
        for ch in chars: # can't use enumerate(chars) since it would start from 0 and override <unk> (LLM-inspired)
            if ch not in self.tokentoidx:
                idx = len(self.tokentoidx)
                self.tokentoidx[ch] = idx
                self.idxtotoken[idx] = ch

        indices = [self.tokentoidx.get(ch, self.tokentoidx["<unk>"]) for ch in text]

        while len(self.tokentoidx) < self.vocab_size:
            stats = self.get_stats(indices)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_idx = len(self.tokentoidx)
            new_token = self.idxtotoken[best_pair[0]] + self.idxtotoken[best_pair[1]]
            self.tokentoidx[new_token] = new_idx
            self.idxtotoken[new_idx] = new_token
            self.merges[best_pair] = new_idx
            indices = self.merge(indices, best_pair, new_idx)

    def encode(self, text):
        unk = self.tokentoidx["<unk>"]
        indices = [self.tokentoidx.get(ch, unk) for ch in text]
        for pair, idx in self.merges.items():
            indices = self.merge(indices, pair, idx)
        return indices

    def decode(self, indices):
        specials = {"<pad>", "<sos>", "<eos>", "<unk>"}
        return "".join(
            self.idxtotoken.get(idx, "<unk>") for idx in indices
            if self.idxtotoken.get(idx) not in specials
        )

    def __len__(self):
        return len(self.tokentoidx)


def build_vocab(pairs, tokenizer): # must be fit on ONLY training data! [made that mistake :>]
    # note: same vocab for both input and output! in other uses (ex translation), requires 2 seperate vocabs
    vocab = Vocab()

    tokens = []
    for buggy_text, fixed_text in pairs:
        tokens.append(tokenizer(buggy_text))
        tokens.append(tokenizer(fixed_text))

    vocab.build(tokens)
    return vocab


class CodeRepairDataset(Dataset): # this must hold the raw text pairs, encode them on the fly, return tensor pairs!! with markers (LLM-inspired)
    def __init__(self, pairs, tokenizer, vocab): # tokenizer can be any str -> list[str] callable, so str.split works for whitespace,  list works for char-level, (LLM-inspired)
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        buggy_text, fixed_text = self.pairs[idx]

        buggy_tokens = self.tokenizer(buggy_text)
        fixed_tokens = self.tokenizer(fixed_text)

        buggy_indices = [self.vocab.tokentoidx["<sos>"]] + self.vocab.encode(buggy_tokens) + [self.vocab.tokentoidx["<eos>"]]
        fixed_indices = [self.vocab.tokentoidx["<sos>"]] + self.vocab.encode(fixed_tokens) + [self.vocab.tokentoidx["<eos>"]]

        return torch.tensor(buggy_indices, dtype=torch.long), torch.tensor(fixed_indices, dtype=torch.long)
    

def batch_padding(batch): # Dataset.__getitem__ returns tensors of different lengths, so we need to pad it before feeding it to our models
    buggy_batch, fixed_batch = zip(*batch) # unzips our batch pair (LLM-inspired)

    buggy_padded = pad_sequence(buggy_batch, batch_first=True, padding_value=0) # assuming idx=0 is <pad>
    fixed_padded = pad_sequence(fixed_batch, batch_first=True, padding_value=0) # it is, since we hard-coded it to vocab

    return buggy_padded, fixed_padded


def get_dataloaders(tokenizer, batch_size=32, max_len=150):
    raw_pairs = load_dataset("google/code_x_glue_cc_code_refinement", "small")

    def extract_pairs(split):
        pairs = []
        for item in raw_pairs[split]:
            buggy, fixed = item["buggy"], item["fixed"]
            # since BPE has not been built yet, we use a dummy tokenizer (say str.split) as a cheap length estimate (LLM-inspired)
            if len(buggy.split()) <= max_len and len(fixed.split()) <= max_len:
                pairs.append((buggy, fixed))
        return pairs

    train_pairs = extract_pairs("train")
    val_pairs   = extract_pairs("validation")
    test_pairs  = extract_pairs("test")

    if hasattr(tokenizer, 'build'):
        all_train_texts = [t for pair in train_pairs for t in pair]
        tokenizer.build(all_train_texts)
        tok_fn = tokenizer.encode
    else:
        tok_fn = tokenizer

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    vocab = build_vocab(train_pairs, tok_fn)
    print(f"Vocab size: {len(vocab)}")

    train_ds = CodeRepairDataset(train_pairs, tok_fn, vocab)
    val_ds   = CodeRepairDataset(val_pairs,   tok_fn, vocab)
    test_ds  = CodeRepairDataset(test_pairs,  tok_fn, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=batch_padding)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=batch_padding)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=batch_padding)

    return train_loader, val_loader, test_loader, vocab