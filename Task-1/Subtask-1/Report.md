# Subtask-1:   Sorted Relative Ranks

This report contains information about my attempts at implementing the Transformer Encoder architecture (and the baselines!)
I have also chosen this sub-task for performing my ablations and analysing the model's behaviours with respect to many parameters/architecture motivations.

## LLM usage
I have commented next to LLM-generated code lines specifically in the source files. LLMs were used to understand concepts like  An LLM was used to generate the diagrams given below. 

## Methodolgy
I decided to make three baselines-
    1. MLP-based model
    2. Vanilla Uni-Directional model
    3. LSTM-based Bidirectional model

After which, I implemented the Transformer Encoder architecture, which I will refer to by 'BERT' for simplicity...\
Given below are diagrams of the architectures used.

### 1. MLP Architecture
<img src=".\\mlp_arch.png" alt="MLP Diagram" width="600" height="500"> 

### 2. RNN Architecture
<img src=".\\rnn_arch.png" alt="RNN Diagram" width="600" height="500">

### 3. biLSTM Architecture
<img src=".\\lstm_arch.png" alt="biLSTM Diagram" width="600" height="500">

### 3. "BERT" Architecture
<img src=".\\bert_arch.png" alt="BERT Diagram" width="600" height="500">

I wanted to gauge the extent to which these models could handle varying length. So I took the raw dataset provided, and sliced them into different lengths (4, 6, 8, and the original 10), creating datasets containing sequences and ranks of lengths 4 to 10. I plan on using these to understand the above.

I have also made my source code modular enough to toggle embeddings and/or normalisations on and off, though I've left this out for the "BERT" model, as I feel it isn't useful for such a complex model to be trained on non-embedded data. 

With that being said, the baseline parameters of each model for the initial evaluations will be as follows...  
* <b>MLP</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`)   
* <b>Vanilla RNN</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`, `layer=1`, `bidirectional=False`)
* <b>biLSTM</b> (`vocab_size=1000`, `sequence_lengths=[4,6,8,10]`, `embed_dim=5`, `layer=1`, `bidirectional=True`)
* <b>BERT</b> (`N=3`, `heads=2`, `dmodel=10`, `dk=5`, `dv=5`, `vocab_size=1000`, `sequence_lengths=[4,6,8,10]`)  

with an AdamW optimizer (`lr=1e-3`, `weight_decay=1e-3`, `betas=(0.9,0.99)`), and a batch_size=256. Also, I will let the `maximum number of epochs` be `2000`. All them will be tested with embeddings initially (which also means no normalisation), with more work done in ablations.

## Baseline Analysis
Following are the training-validation loss history of the models, for sequence_length=10.  
MLP Model RNN Model biLSTM Model
<img src=".\\Ablation data\\Baseline analysis\\10_mlp_loss.png" alt="MLP loss history" width="300" height="250">
<img src=".\\Ablation data\\Baseline analysis\\10_rnn_loss.png" alt="MLP loss history" width="300" height="250">
<img src=".\\Ablation data\\Baseline analysis\\10_lstm_loss.png" alt="MLP loss history" width="300" height="250">
