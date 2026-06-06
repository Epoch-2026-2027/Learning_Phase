# Subtask-3: Poetry Generation --> Decoder-only Transformer

## Overview

This README explains two notebooks in the downloads folder:

- `Decoder_Only (2).ipynb` — a TensorFlow implementation of a decoder-only transformer trained on the `merve/poetry` dataset.
- `BaselineModels_3 (4).ipynb` — a PyTorch notebook that implements baseline sequence models including a bigram model, gMLP, vanilla RNN, and LSTM for the same poetry data.

---

## Purpose

The goal of these notebooks is to compare:

- simple baselines for next-token modeling (`bigram`, `RNN`, `LSTM`, `gMLP`)
- a modern autoregressive sequence model (`decoder-only transformer`)

Both notebooks are trained on the same poetry dataset, with the same preprocessing strategy and vocabulary conventions. This makes it easier to compare how recurrent models, gating models, and self-attention models behave on the same task.

---

## Dataset and Preprocessing

Both notebooks use the Hugging Face dataset `merve/poetry`.

Key preprocessing steps:

1. Load the dataset and convert the training split to a pandas DataFrame.
2. Add special sentence markers:
   - `<s>` at the beginning
   - `</s>` at the end
3. Normalize the text:
   - replace Windows line breaks (`\r\n`) with spaces
   - convert to lowercase
   - add spaces around punctuation such as `, ; . ! ?`
   - collapse multiple whitespace characters into a single space
4. Split into tokens using `str.split()` on spaces.

This creates a single token sequence over the entire dataset, then the models build sentence-level examples from that token stream.

---

## Tokenizer

Both notebooks define a simple custom tokenizer with four reserved tokens:

- `<PAD>` = 0
- `<UNK>` = 1
- `<s>` = 2
- `</s>` = 3

`Decoder_Only (2).ipynb` uses a capped vocabulary of `max_vocab=5000`. (only using the 5000 most common words, this was so that the model wouldn't waste time memorising random insignificant words)
`BaselineModels_3 (4).ipynb` builds the full vocabulary from all tokens in the dataset.

The tokenizer provides:

- `fit()` — build the vocabulary from token frequencies
- `encode()` — convert tokens to integer IDs
- `decode()` — convert integer IDs back to tokens

This is enough for both training and generation.

---

## Baseline Models in `BaselineModels_3 (4).ipynb`

### Bigram Model

The bigram model is a simple frequency-based baseline:

- count every adjacent token pair `(w1, w2)` across the corpus
- remove transitions from `</s>` to `<s>` so sentence boundaries are not joined
- sample the next word from the conditional distribution given the current word

This is a non-learned baseline that illustrates how much structure can be captured by local token transitions alone.

### gMLP

The gMLP model is a modern alternative to a transformer that uses spatial gating instead of full attention.
I used this because I didn't want to just implement a simple MLP model, has been done multiple times before.

Key components:

- `SGU` (Spatial Gating Unit)
- `gMLPBlock` with
  - LayerNorm
  - dense expansion
  - GELU activation
  - spatial gating
  - residual connection
- `gMLP` model with
  - token embedding
  - learned positional embeddings
  - stacked gMLP blocks
  - final layer normalization and output projection


### LSTM

The LSTM model is a classic recurrent sequence model.

Structure:

- embedding layer with padding support
- stacked LSTM layers
- dropout
- residual projection from embedding to LSTM outputs
- final linear projection to vocabulary logits


### Vanilla RNN

A simpler recurrent baseline.

Structure:

- embedding layer
- vanilla `nn.RNN`
- linear projection to vocabulary logits

This model is smaller and more constrained than the LSTM, and therefore serves as a useful comparison for the transformer.

---

## Decoder-Only Transformer in `Decoder_Only (2).ipynb`

### Architecture Components

#### Token Embedding

The model starts with an embedding layer that converts token IDs to dense vectors of size `d_model`.

#### Positional Encoding

Transformers are position-agnostic by default because they process every token in parallel, so the notebook adds a `PositionalEncoding` layer using sine and cosine functions.

This layer produces a fixed positional embedding of shape `(1, max_len, d_model)` and adds it to the token embeddings.

This signals the model where each token occurs in the sequence.

#### Causal Masking

The transformer uses a causal mask so each position can only attend to previous positions and itself.

The mask is computed as:

- `1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)`

This produces a mask with ones above the diagonal, which is then added to attention scores.

When the mask is applied, future positions receive large negative logits and are effectively ignored by softmax.

#### Multi-Head Self-Attention

The `MultiHeadAttention` layer performs the core transformer operation:

- project inputs into query, key, and value spaces
- split into multiple heads
- compute scaled dot-product attention for each head
- apply the causal mask
- combine heads back into one tensor
- project the result to `d_model`

Using multiple heads allows the model to attend to different aspects of prior history in parallel.

#### Decoder Block

Each `DecoderBlock` is built from:

- multi-head attention
- dropout
- residual connection + layer normalization
- two-layer MLP (`ff1`, `ff2`) with ReLU activation
- another residual connection + layer normalization

This mirrors the standard transformer decoder block, but only with the decoder stack, since there is no encoder.

#### Full Transformer Model

The `DecoderTransformer` model contains:

- token embedding
- positional encoding
- dropout
- a stack of `DecoderBlock`
- final layer normalization
- final dense projection to vocabulary logits

During the forward pass:

1. embed the input tokens
2. add positional encoding
3. apply dropout
4. process through each decoder block with the causal mask
5. normalize and project to logits

This produces logits for every position, and the loss compares each predicted token against the true next token.

---

## Training Setup

### Autoregressive next-token learning

The model is trained to predict token `t+1` from tokens `0..t`.

For each training example:

- input: `[<s>, token_1, token_2, ..., token_{n-1}]`
- target: `[token_1, token_2, ..., token_{n-1}, </s>]`

(standard autoregressive training objective used by decoder-only architectures)

### Gradient clipping

Gradients are clipped by global norm to `1.0` to stabilize training.

This is especially helpful for deep sequence models, where unbounded gradients can cause training instability.

---

## Evaluation and Metrics

Both notebooks use the Hugging Face `evaluate` package with the `accuracy` metric.

Evaluation strategy:

- flatten model predictions and references
- mask padding tokens
- compute accuracy only over real tokens

This gives a **token-level accuracy** score for next-token prediction.

`BaselineModels_3 (4).ipynb` also computes **perplexity** for one of the models by summing cross-entropy over non-pad tokens and exponentiating.

---

## Generation / Demo

### Decoder-only transformer generation

`Decoder_Only (2).ipynb` shows how to generate text greedily from a prompt:

1. start with an initial token list, e.g. `['the']`
2. encode it with the tokenizer
3. pad to `max_len`
4. run the model and take `argmax` of the last logit
5. append the predicted token and repeat
6. stop when `</s>` is generated or the sequence reaches `max_len`

This is the canonical autoregressive decode loop for decoder-only models.

### Baseline model generation

`BaselineModels_3 (4).ipynb` also shows **greedy decoding** for:

- gMLP
- RNN
- LSTM

and compares their predicted next-token distributions.

---

## Visualizations in the Transformer Notebook

`Decoder_Only (2).ipynb` includes two interpretability visualizations:

- attention heatmap for a user-provided phrase
- top predicted token probabilities for the next token

These help illustrate what the model is attending to and what it believes are the most likely next words.

---

## Decoder-Only Transformer's Significance

The decoder-only transformer differs from the baseline models in several important ways:

- no recurrence: it processes the entire input sequence with self-attention rather than step-by-step recurrence
- causal masking: it only sees past tokens, which matches the autoregressive generation task
- parallel computation: training can process all positions in a sequence simultaneously
- long-range context: attention can directly connect distant tokens without passing through a recurrent state
- flexible representation: multi-head attention lets the model learn multiple ways to compare past tokens

This architecture is the foundation for GPT-style language models, and the notebook shows a compact version of that idea using only a few decoder blocks.

---


