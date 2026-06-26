# &#x09;	    SUBTASK - 1

### &#x09;	     Array Element Ranking

##### \-> Methodology

&#x20;  

&#x20;   I methodology was quite simple . I tried to experiment between MLP vs LSTM vs Encoder-Only Transformer and see how they are working.

&#x20;   I tried to keep the dimension high so that the model gets more parameters to learn and predict better . I projected the input into a 

&#x20;   higher dimension for the transformer model and introduced non - linearity into it so that it can learn more positional information and 

&#x20;   has more parameters to learn.

##### 

##### \-> Architectural Choices

&#x20;   1. MLP model

&#x20;      Trained a simple neural network which has two linear layers and a relu layer which are connected in the below way

&#x20;      

|                                       Linear layer -> ReLU layer -> Linear Layer|
|-|

&#x20;      It has a very few number of parameters but still managed to capture 54.6% of accuracy.

&#x20;  

&#x20;   2. LSTM Encoder Decoder Seq2Seq model

&#x20;      This is a basic encoder decoder lstm model , In which Input is passed sequentially to encoder and the decoder is initialized with

&#x20;      the hidden , cell states of the encoder and the decoder's final hidden state is projected to the target dimension to predict 

&#x20;      probabilities.

|                                  Input -> Encoder LSTM -> Decoder LSTM -> Linear layer                           |
|-|

&#x20;    

&#x20;    3. Encoder - Only Transformer

&#x20;       This is the best model out of all the three I made . It uses bi-directional self attention to find the relation between token to

&#x20;       token which was very useful. Again a basic normal architecture , but this tested a lot of patience of mine and helped me learn a 

&#x20;       lot which helped in my next subtask . I mentioned my architecture below in the box.

&#x20; 

|Input -> Projection into higher dim -> positional encoding -> Bi-directional Self Attention -> Residual connection -> Feed forward projecting into output dim|
|-|

&#x20;       

&#x20;       It resulted an accuracy of 87.37% accuracy which is the best.



##### \-> Numerical Representation Strategy

&#x20;   I tried by "passing the inputs directly "," projecting inputs using a linear function" , "projecting them using a non-linear" function

&#x20;   In all the three the last method worked so good where as the other two didn't help at all while predicting they resulted in the same 

&#x20;   prediction ranks for each and every value which is so bad.



##### \-> Experiments done

&#x20;    1. Trying all the basic models to find the difference between the way they are predicting

&#x20;    2. Tried to increase the layers by passing the predictions to the same model again ehich is literally of no use.

&#x20;    3. Trying to pass inputs in different ways and finding how they are effecting the outputs.



##### \-> Attention Visualizations

<img width="790" height="700" alt="Attention_heatmap" src="https://github.com/user-attachments/assets/6f4f12c1-6eab-4022-a352-95f34395ce29" />


&#x20;   The X\_axis is keys and Y\_axis is queries . This matrix says how each word is related to other by using purely maths , somethings like

&#x20;   this seem amazing to me.



##### \-> Conclusion

<img width="857" height="701" alt="image" src="https://github.com/user-attachments/assets/44fcffac-52bd-44f5-bf63-74cb46f2f538" />


&#x20;    As you can see the losses are decreasing means the model is good and not overfitting and just enough to predict things good.

&#x20;    In my transformer model I didn't increase the layers and also didn't use multi head attention I saved both of them for the next task 

&#x20;    Anyway here I don't see more depth of layers much useful , It may help my model to incerase a 8-10% of accuracy which is OK.



&#x20;    TQ for all the heads who helped me in making this task be atleast OK OK like this . I learned a lot in my first task itself with a 

&#x20;    lot of unexpected experiments and new terminologies. Atlast I want to say about my LLM usage and Google usage . I used self searched 

&#x20;    medium articles to know how to code these transformers and LLM was mostly used for error debugging and reshaping of tensors and loss 

&#x20;    logits . Thank you once again. By the way demo model is in my colab notebook which I submitted.

