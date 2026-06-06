# BaselineModels_2 Notebook — Seq2Seq Code Refinement (RNN & LSTM)

## Overview

This README describes the workbook saved as [Downloads/BaselineModels_2 (2).ipynb](Downloads/BaselineModels_2%20(2).ipynb). The notebook implements a small, self-contained sequence-to-sequence experiment that trains two kinds of encoder–decoder models on a code-refinement dataset (buggy -> fixed code): a SimpleRNN-based seq2seq and an LSTM-based seq2seq with additive attention.

## Purpose

Explain and reproduce the steps to train simple seq2seq models for code change prediction using TensorFlow Keras. The notebook focuses on:
- Building a minimal `Tokenizer` and preparing (buggy, fixed) token pairs.
- Creating a `batches()` generator that pads inputs and produces encoder/decoder inputs and decoder targets.
- Defining and training two models:
  - `rnnencoder` / `rnndecoder` — SimpleRNN encoder + decoder (single-directional)
  - `encoder` / `decoder` — Bidirectional LSTM encoder and LSTM decoder with `AdditiveAttention`
- Evaluating using the `evaluate` package's `accuracy` metric and plotting loss/accuracy curves.

## Notebook Sections 

- Tokenizer class — a small vocabulary builder with `fit()`, `encode()`, and `decode()` methods. Adds tokens: `<PAD>`, `<UNK>`, `<s>`, `</s>`.
- `prepare(df, tokenizer)` — wraps each string with sentence tokens, splits into word tokens, and encodes pairs into integer sequences.
- `batches(pairs, batch_size)` — yields padded `enc_inputs`, `dec_inputs`, and `dec_targets` as `tf.constant` tensors. Uses `0` as the padding/ignore class.
- SimpleRNN model (RNN section) — builds `rnnencoder` (returns state) and `rnndecoder` (accepts initial state), trains with a custom loop using `tf.GradientTape`.
- LSTM model (LSTM section) — builds a bidirectional LSTM encoder that returns per-step outputs and concatenated final states; decoder uses an LSTM followed by `AdditiveAttention` and a final `Dense` to project to the vocabulary.
- Training loops — per-epoch loops for both models that:
  - accumulate `train_loss` and append average to `losstrack`
  - compute validation accuracy with `evaluate.load("accuracy")` and append to `acctrack`
  - print progress per epoch
- Plotting — uses `matplotlib` to visualize loss and validation accuracy across epochs for each trained model.
- Testing — runs the trained models on small test slices and prints `Test Acc`.

## Implementation Details

- Padding token id: `0` is used as the padding / `ignore_class` value. The loss is constructed with `ignore_class=0` when creating `SparseCategoricalCrossentropy`.
- Loss function: `keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)`.
- Optimizer: `keras.optimizers.Adam(learning_rate=0.001)`.
- Metric: `evaluate` package `accuracy` is used. The notebook flattens predictions and targets, masks out padding, and feeds remaining tokens into the metric.
- Dataset slices: training/validation/test slicing is done with Python indexing (e.g., `encoded[:2500]`, `valenc[:1000]`, `testenc[:100]`) — adapt these ranges if you want to use more data or a smaller subset.

## Hyperparameters (as used in the notebook)

- RNN section: `batchsize = 64`, `d_model = 256`, `EPOCHS = 25`
- LSTM section: `batchsize = 100`, `d_model = 128`, `EPOCHS = 25`

## Visual Representation

- RNN loss/accuracy figure: ![rnn](output21.png)
- LSTM loss/accuracy figure: ![lstm](output22.png)


## Architecture Analysis

- **Bidirectional encoders:** Bidirectional encoders provide context from both past and future tokens when producing encoder representations. For a code-refinement or sequence-mapping task, they improve the encoder's ability to represent tokens that require context on both sides which usually raises accuracy.

- **Deeper recurrent networks:** Adding recurrent layers increases model capacity and can learn more complex mappings, but depth also makes optimization harder and increases training time. When using deeper RNNs, it was only effective with residual connections, layer normalization, etc.

- **Different embedding dimensions:** Larger embedding dimensions let the model represent richer lexical and syntactic information, improving performance when data supports it. Smaller embeddings reduce parameters and overfitting risk. 

- **Attention mechanisms:** Attention enables the decoder to focus on relevant encoder time-steps and supports global comparisons across the input sequence. For tasks like code edits, attention usually yields the largest improvement versus vanilla encoder–decoder RNNs.

- **Dropout:** Helps regularization. Dropout between recurrent/dense layers reduces overfitting, especially for smaller datasets. 

- **Gradient clipping:** Helps to stabilise. Clipping (by norm or value) prevents exploding gradients in recurrent models and makes training more stable, allowing higher learning rates and fewer training crashes.

- **Different tokenization strategies:** For code, subword or identifier-aware tokenization often provides the best balance, considering that so much of it is repeated over and over again.




