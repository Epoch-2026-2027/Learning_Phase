## &#x20;                      SUBTASK - 2

### &#x20;                        CODE REFINEMENT



##### \-> Methodology

&#x20;   I tried to experiment from basic to complex models , and tried to understand how they work . Choosing higher dimension and good tokenization techniques for a better output and sticking to the base models and trying to explore them the most.



##### \-> Architectural choices

&#x20;    1. LSTM Seq2seq single layer model :

&#x20;       This is a normal basic model which consist of encoder , decoder and a seq2seq class with a token-wise level accuracy

|    Input -> Embedding Layer -> Encoder LSTM -> Decoder LSTM (gets input token) -> Fully connected layer to output dimension|
|-|



&#x20;    2. LSTM Seq2Seq multi layered model :

&#x20;       This is a more generalization of above model this has a increase in number of LSTM layers , which will try to capture more complexity in the model . This also follows above order only but has 4 Encoder LSTM layers and 4 Decoder LSTM layers

&#x20;  

&#x20;    3. Encoder Decoder Transformer model:

&#x20;       This is also a more complex baseline model which uses Multi Head Attention , I used it here because I used single head attention 

&#x20;       in my before task .

|           Encoder Input -> Embedding and positional encoding -> Multi Head Attention -> Residual Connection|
|-|
|Decoder Input -> Embedding and Positional encoding -> Self masked attention -> Residual connection -> Encoder Decoder attention -> Fully connected layer to output dimension|





##### \-> Experiments Done

&#x20;   1. Not much experiments in this task the reason is my large dataset and my laptop capacity , I was unable to access GPU so all the

&#x20;      training was done on CPU which took literally 40 min for 5 epochs of learning so my main focus was on model building and 

&#x20;      understanding how did all this make sense and how are they predicting rather than concentrating more on model complexity.

&#x20;   2. I tried to build a tokenizer of my own.



##### \-> Conclusion :

&#x20;    Very less to write in this report from my side since I didn't do anything much . All the time I spent was on debugging and taking care that I understood everything I wrote and I don't have any data leakage to my model. But still I to think that I should have spent more time on this model and try to examine what is the issue . I am pasting my training and validation losses down there . The main task is to understand the architecture I think I did that part well . Thank you for all the heads for helping me in this task.

<img width="875" height="701" alt="image" src="https://github.com/user-attachments/assets/779c71d1-a60e-47a6-9924-73b20e67b7d9" />




##### \-> LLM Usage :

&#x20;   I used LLM in this model to mostly debug the issue and discuss logics with it " why my route is correct? " and to code the masking 

&#x20;   mechanism in for the teacher forcing and decoder part . I wrote most of the code in it except those parts by learning them beforehand 

&#x20;   from medium articles and other resources kept by you .

