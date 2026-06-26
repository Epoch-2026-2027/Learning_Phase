## &#x20;                 ROAD SEGMENTATION TASK



##### \-> Training Pipeline:

###### &#x20;    1. U-Net Model:

&#x20;          

|Encoder:|
|-|
|Channel Flow : 3 -> 64 -> 128 -> 264 -> 512 -> 1024|
|Size Flow : 1500^2 -> 750^2 -> 375^2-> 187^2 -> 93^2|



In Bottle Neck , there will be a convolution layer which continues the same number of channels



|Decoder:|
|-|
|Channel Flow : 1024 -> 512 -> 264 -> 128 -> 64 -> 1(binary layer predicting one or zero)|
|Size Flow : 93^2 -> 187^2 -> 375^2 -> 750^2 -> 1500^2|



I used padding , that is the reason that size of 93 changed to 187 instead of 186.

All the layers are designed based on the model in the original paper.



###### &#x20;   2. Attention U-Net Model:

&#x20;        This also has the same flow of size and channels , the only change was in the skip connections which used Attention gates to capture the relation between the lower layer of decoder and the corresponding layer in encoder , capturing both spatial and masking information.

!\[Image of Attention block]





##### \-> Preprocessing Pipeline:

&#x20;   I didn't do anything as part of this the only thing I did is , I reduced the number of images used for training to remove the OOM error which occurred , I thought not to resize the image to 512\*512 since I may loose helpful info and many roads which need to be identified



##### \-> Evaluation :

&#x20;   I used BCE + Dice Loss , BCE loss to find the loss of predicting the mask and Dice loss to correctly find the intersection region of road we predicted and actual part of road. Dice Loss uses IoU to find the correct intersection.



##### \-> LLM Usage:

&#x20;   1. In this task I mostly used LLM to solve my OOM error , which I was facing first time and don't know what to do , I also consulted heads regarding this issue they gave me few ideas and I took help of LLM to find the implementation.

&#x20;   2. I also read several articles from google regarding their implementation and reasons how and why they are working.



##### \-> Conclusion:

&#x20;    As last I am attaching my validation vs training loss , I only trained for 5 epochs but still it gave a very good result. Visualizations of the road segmentation are in the colab notebook.

!\[Losses image]

