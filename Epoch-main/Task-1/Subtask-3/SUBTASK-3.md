# &#x09;	      SUBTASK-3

## &#x09;	     POETRY GENERATION



##### \-> METHODOLOGY:

&#x20;   I tried new things this time exploring new baseline models , evaluation metrics , depth analysis and tokenization . I made a bigram

&#x20;   model , which I made using the basic principles and completely understanding how it works and it's flaws. Then I made a decoder only 

&#x20;   transformer model which is the first time I made a model with 3 layers of depth and also explored a new evaluation metric and tried to

&#x20;   understand how it works.



##### \-> ARCHITECTURAL CHOICES:

&#x20;   1. Bigram model :

&#x20;   This is a basic text book model made by me , I got to learn about this model right now , I tried to understand it's flaws and it's

&#x20;   logic . I made a look up dictionary which has the counts of the next word which can occur based on the previous word . Later randomly

&#x20;   chose a next word which is weighted with respect to counts , that showed a better response with 8 - 9 % accuracy.

&#x20;  

&#x20;   2. Decoder Only Transformer :

&#x20;   This also has a basic architecture but with a depth of 3 layers for capturing the better complexity in the dataset.

&#x20;   

|Embedding Layer -> Positional Encoding -> (Multi Head Attention Layer -> Residual Connection and norm -> Feed forward Layer -> Residual Layer and norm)\*3 -> Projecting to output dimension i.e., vocab size|
|-|

&#x20;  

&#x20;   I have also tried perplexity metric for the first time and I really loved the way it said how it said the accuracy .



##### \-> EXPERIMENTS CONDUCTED:

&#x20;    1. Tried different tokenization methods

&#x20;    2. Tried different evaluation metrics

&#x20;    3. Tried to find the importance of Depth of Layers



##### \-> Evaluation Metrics:

&#x20;    1. Token level evaluation for bigram model

&#x20;    2. Perplexity for Decoder only transformer model



##### \-> MODEL BEHAVIOUR :

&#x20;   Both the models learnt very well , but the transformer model took a lot of time to iterate due to many number of sequences.



##### \-> CONCLUSION :

&#x20;   I am attaching my Epoch Losses here so that we can get an idea of how the model learnt

<img width="848" height="707" alt="image" src="https://github.com/user-attachments/assets/668ce23c-3142-4e22-a510-90163662d934" />

&#x20;   Thank you for all the heads for helping me learn.



##### \-> LLM USAGE:

&#x20;    I mostly used LLM for debugging and for research about the model and evaluation techniques, all the architectural decisions and 80% of the code is written by But by these tasks I got to learn mainly how to handle the shapes of the tensors

