### &#x20;                     Diffusion Models Task



##### \-> Architectural Description:

&#x20;   In this part I will present you how the data was flowing in the architecture.

|Input Image -> Corrupting the image partially using scheduler -> Predicting the noise per timestamp and removing the noise -> Passing the output image through ResNet50 arch to get predictions.|
|-|



This is different from the actual work of a diffusion model , because in this task since we were given the labels also I thought that we can predict the labels of the image made by diffusion models.



##### \-> Experiments and Ablation Studies:

&#x20;    1. I experimented with the timestamps, First I corrupted the image fully then tried to generate the same image which I got to know is not how diffusion models work , they generate image from random noise , so to predict that I corrupted it only 20% and denoised it so that it can generate same image instead of generating a image which I don't know what is it even though it's correct. So first I only got 9-10% accuracy , after making the change the accuracy jumped to 28-29%.

&#x20;    

&#x20;    2.Using different schedulers:

&#x20;      I used DDPM and DDIM schedulers as part of ablation , I properly don't know why DDIM worked better , but it was able to predict the label with a accuracy of 63-64% which I thought was great.

&#x20; 

&#x20;    3. I tried different batch sizes and a few of them resulted in OOM error, so I had to stick to lower batch size and I also reduced the number of images so that the training takes place faster and I think this didn't affect the predictions more. 



##### \-> Model Behaviour:

&#x20;   Since both unet and schedulers are pretrained , the only part I was training is the last layer of classifier layer which predicted the classes using the features it has , I would say the diffusion model was performing good as I have also visualized the original image and the generated image by a diffuser model in my colab notebook they both looked almost same. The main issue I got was memory management and normalization problems since I used the weights of Image-Net classifier I had to normalize it in that way for which I took help of LLM.



##### \-> LLM Usage:

&#x20;    1. I used LLM to find the way of extraction of models which I need according to the task.

&#x20;    2. I used LLM for reshaping of the images using transforms in torchvision module which was so helpful.

&#x20;    3. I used the most of LLM help at debugging by OOM error , other than this all the architectural decisions are made by me just I had to lookup a few implementations.



##### \-> Conclusion:

&#x20;    I have visualized the images of original and generated images , I have also shown my predictions list and I think everything is clearly commented. Thank you for the heads who helped me.

