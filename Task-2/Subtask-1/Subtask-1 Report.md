# <u>SubTask-1: Road Segmentation from Satellite Images</u>
In this report, I lay down my approaches and findings relating to binary image segmentation, for the below mentioned problem statement and dataset. Models were trained and inference was carried out while keeping in mind analysing quality and computational limits.

## <u>LLM Usage</u>
Any line that was directly pulled from an LLM has a comment next to it for the same. 

## <u>Problem Statement</u>
*"Given a satellite image as input, your model must predict a pixel-wise mask indicating the locations of roads."*

-Which is a Semantic Segmentation task.

## <u>Dataset Exploration</u>
For this sub-task, I will be using the [DeepGlobe 2018 Road Extraction Challenge Track dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset/data). It contains its own training-validation-testing split, but unfortunately only the training split contains the ground truth masks. So I will be be using the given training split, for my own training-validation-testing split-ups.

The training split contains ~6.3k datapoints of satellite image - mask pairs, each image being 1024x1024 pixels. For easier training and inference, I will only be using 1,500 datapoints for my split-ups, that too downscaled to 256x256 images.

To further visualize the data, I decided to focus on two metrics in each image - mean road thickness in pixels, and average road mask density.
1. Mean Road Thickness (in px) - Calculated by thresholding the masks, then comparing the distance of the skeletal lines of the roads to the nearest `o` pixel. This has been calculated over the entire image for each image and averaged image-wise.
<p align="center">
	<img src="subtask1 report images/_road_thickness_distribution.png" alt="road_thickness_distribution" style="width:60%; height:auto;">
</p>

2. Average Road Mask Density - Calculated by getting the fraction of non-zero pixels in the mask, divided by `Height x Width`.
<p align="center">
	<img src="subtask1 report images/_average_road_density_distribution.png" alt="average_road_density_distribution" style="width:60%; height:auto;">
</p>

Clearly, we can see that images with (on average) narrow roads and density dominate the dataset. The issue is that depending on how training is conducted, this can lead to the model learning almost nothing, or not being able to generalize precisely.

## <u>Methodology</u>
The original DeepGlobe dataset was first sliced to procure 1,500 datapoints, which were then downscaled to 256x256 images of satellite-mask pairs. A first run went through and calculated the mean and standard deviation of all the images for RGB channels. Using the acquired mean and standard deviation, the images were Z-Normalized, before being saved as NumPy binary files (`.npy`).
The entire pre-processing script, and the ensuing dataloader script, were both implemented lazily to conserve precious computing resources.  

*<i>The original Kaggle dataset was processed by a python script `organize.py` into a folder named "CV_Data", with "sats" and "masks" folders.</i>

*<i>The pretrained checkpoints were to big to upload, so I've included them in a google drive folder - https://drive.google.com/drive/folders/1otdFdcYK9oZs7lsLK6fSDCsWNtmNv7Tj?usp=sharing</i>

The baseline model was then established - <u>a classical U-Net model</u>. To compare potential improvements with it, the following models were chosen-
1. <u>U-Net with ResNet backbone </u>
	(to see how much faster and better it would train and generalize with a pretrained backbone)
2. <u>U-Net with Attention Gates</u>
	(to see how attention would fare in semantic segmentation)
3. <u>U-Net with Attention Gates and Resnet backbone</u>
	(to see how adding attention on top of a backbone would improve the model)
4. <u>SegFormer</u>
	(to verify if a transformer model would out-compete the other models)

I did not want to dive into `FPNs`  and `DeepLabV3`, because
1. FPNs seemed tempting due to their computational efficiency. However, they seemed to perform less than U-Net based models in the actual DeepGlobe challenge. So I held off of them for this reason.
2. The exact opposite problem applied to DeepLabV3. Training would have taken a lot of time, thus I took the option to exclude it from this task.

*\*For the SegFormer, I initially wanted to train it from scratch. But after seeing lackluster results from it, I decided to give important to the pre-trained version that was then fine-tuned on the dataset. I will mention my results on this in the section for the SegFormer.*

From the Dataset Exploration, it is clear that the dataset is heavily weighted towards samples with narrower roads and lesser road density. Keeping this in mind, I will be using `Focal Loss` as my baseline, since Focal Loss is suited for problems were the model can get `easy validation` on background pixels.
Other losses and loss combinations will be experimented with in the Experiments section.

For all the models, I will be using the same training hyper-parameters-
`Max Epochs=50` `Learning Rate=2e-5 (with scheduled sampling of 0.8 every 10 epochs)`
`AdamW optimizier with Weight Decay=1e-3` `Batch Size=2`

During inference, a suitable threshold is used to first convert the model predictions into binary masks, which is then tested using two metrics:-
1. <u>Dice-Sørensen coefficient</u>
2. <u>IoU Score</u>

## <u>Baseline Model</u>
#### <u>Classical U-Net Architecture (with Skip Connections)</u>
<p align="center">
	<img src="subtask1 report images/Pasted image 20260624191259.png" alt="average_road_density_distribution" style="width:60%; height:auto;">
</p>
The model was implemented almost exactly as in the [original paper](https://arxiv.org/abs/1505.04597) describing it, with the only differences being the input image dimensions being 256x256, and that padding was used to ensure that the output image's dimensions matched the mask dimensions.

**<u>Training Loss:</u>**
<p align="center">
	<img src="subtask1 report images/classicunet_model_loss_ 1.png" alt="Training Loss History" style="width:50%; height:auto;">
</p>

**<u>Evaluation and Inference:</u>**
<p align="center">
Average IoU score = 0.1789 | Average Dice score = 0.2874
	<img src="subtask1 report images/classicunet_model_predictions.png" alt="classicunet_model_predictions" style="width:60%; height:auto;">
</p>


## <u>Model Improvements</u>
#### **<u>1. Attention U-Net</u>**
Our first model improvement is by introducing soft attention, by adding Attention Gate mechanisms to the skip connections as described in [this article](https://medium.com/data-science/a-detailed-explanation-of-the-attention-u-net-b371a5590831). Below is the diagram for the attention gate.
<p align="center">
	<img src="subtask1 report images/Pasted image 20260624193044.png" alt="attention gate diagram" style="width:60%; height:auto;">
</p>
It is similar to "Cross Attention", allowing the model to make unaligned weights smaller, while making aligned weights bigger.

**<u>Training Loss:</u>**
<p align="center">
	<img src="subtask1 report images/attendedunet_model_loss_ 1.png" alt="attendedunet_model_loss_" style="width:60%; height:auto;">
</p>

**<u>Evaluation and Inference:</u>**
<p align="center">
Average IoU score = 0.1708 | Average Dice score = 0.2803
	<img src="subtask1 report images/attendedunet_model_predictions.png" alt="attendedunet_model_predictions" style="width:60%; height:auto;">
</p>

#### **<u>2. U-Net with ResNet Backbone</u>**
In this parallel implementation, the encoder has been replaced with a pretrained ResNet-18 model, which serves as the feature extractor. The skip connections now originate from the corresponding level of the ResNet-18 architecture.
<p align="center">
	<img src="subtask1 report images/Pasted image 20260624205508.png" alt="resnetunet" style="width:60%; height:auto;">
</p>
*<i>The above diagram is accurate upto the last step, where I did not do another double convolution for 512-->1024</i>

**<u>Training Loss:</u>**
<p align="center">
	<img src="subtask1 report images/resnetunetFOCALLoss_model_loss_.png" alt="resnetunet_loss" style="width:60%; height:auto;">
</p>

**<u>Evaluation and Inference:</u>**
<p align="center">
Average IoU score = 0.3025 | Average Dice score = 0.456
	<img src="subtask1 report images/resnetunet_model_predictions.png" alt="attendedunet_model_predictions" style="width:60%; height:auto;">
</p>

#### **<u>3. Attention U-Net with ResNet Backbone</u>**
This implementation combines both soft-attention and ResNet-18 backbone.

**<u>Training Loss:</u>**
<p align="center">
	<img src="subtask1 report images/resnetattendedunet_model_loss_.png" alt="resnetunet" style="width:60%; height:auto;">
</p>

**<u>Evaluation and Inference:</u>**
<p align="center">
Average IoU score = 0.1926 | Average Dice score = 0.3148
	<img src="subtask1 report images/resnetattendedunet_model_predictions.png" alt="resnetattendedunet_model_predictions" style="width:60%; height:auto;">
</p>


#### <u>4. SegFormer - A Transformer-based segmentation model</u>
Here, SegFormer architecture has been implemented to gauge the effectiveness of a transformer-based segmentation model compared to the other CNN based networks.
<p align="center">
	<img src="subtask1 report images/Pasted image 20260624222644.png" alt="segformer_architecture" style="width:70%; height:auto;">
</p>
Here, I initially tried to train one from scratch. But it offered poor results, most likely due to the limitations I applied on the data pool. I tried a pre-trained version too, that I hoped would perform better when fine-tuned.

**<u>Training Loss:</u>**
<div align="center"> <table> <tr> <td align="center"><b>SegFormer (Pretrained)</b></td> <td align="center"><b>SegFormer (Scratch)</b></td> </tr> <tr> <td><img src="subtask1 report images/segformer_model_loss_.png" width="340"/></td> <td><img src="subtask1 report images/scratchsegformer_model_loss_.png" width="340"/></td> </tr> </table> </div>

**<u>Evaluation and Inference:</u>**
<div align="center">
<table>
<tr> 
<td align="center"><b>SegFormer (Pretrained)</b></td> 
<td align="center"><b>*SegFormer (Scratch)</b></td>
</tr>
<tr> 
<td align="center">Average IoU score = 0.2381</td> 
<td align="center">Average IoU score = 0.0707</td>
</tr>
<tr> 
<td align="center">Average Dice score = 0.3742</td> 
<td align="center">Average Dice score = 0.1258</td>
</tr>
<tr> 
<td><img src="subtask1 report images/segformer_model_predictions.png" width="340"/></td>
<td><img src="subtask1 report images/scratchsegformer_model_predictions.png" width="340"/></td>
</tr> 
</table> 
</div>

## <u>Experiments</u>
Now, experimenting with all the loss functions and combinations.
1. <u>Binary Cross Entropy Loss (BCE)</u>
	Loss = {$l_1,....,l_n$}$^T$
	$l_n​=−w_n​[y_n​⋅logσ(x_n​)+(1−y_n​)⋅log(1−σ(x_n​))]$

2. <u>Dice Loss</u>
	Loss = $1-\frac{2 \sum_{i=1}^{N} P_i T_i}{\sum_{i=1}^{N} P_i^2 + \sum_{i=1}^{N} T_i^2}$ 
	where $P_i$ and $T_i$ represent the Predicted Mask and the corresponding Ground Truth.

3. <u>Focal Loss</u>
	Loss = $-\alpha_{t}(1 - p_{t})^{\gamma}\log(p_{t})$ 
	where, $\alpha_t$ is the weighting factor, $\gamma$ is the modulating factor,
	and $p_t$ is the probability of the predicted label ($p$ for foreground and $1-p$ for background) 
	$\alpha_t$=0.25 and $\gamma_t$=2 were used, same as the original RetinaNet paper the loss function was introduced in.
 
In addition to these single loss functions, two combinations were also evaluated-
-  BCE + Dice Loss 
-  BCE + Dice + Focal Loss

For experimentation purposes, I will be using the same training hyper-parameters, and the Res-Net enabled U-net model.

<div align="center">
<table>
<tr> 
<td align="center"><b>BCE Loss</b></td> 
<td align="center"><b>Dice Loss</b></td>
</tr>
<tr> 
<td align="center">Average IoU score = 0.3527</td> 
<td align="center">Average IoU score = 0.3683</td>
</tr>
<tr> 
<td align="center">Average Dice score = 0.5123</td> 
<td align="center">Average Dice score = 0.5266</td>
</tr>
<tr> 
<td><img src="subtask1 report images/resnetunetBCELoss_model_predictions.png" width="340"/></td>
<td><img src="subtask1 report images/resnetunetDICELoss_model_predictions.png" width="340"/></td>
</tr> 
</table> 
</div>
<div align="center">
<table>
<tr> 
<td align="center"><b>BCE + Dice Loss</b></td> 
<td align="center"><b>Focal Loss</b></td>
</tr>
<tr> 
<td align="center">Average IoU score = 0.1171</td> 
<td align="center">Average IoU score = 0.3025</td>
</tr>
<tr> 
<td align="center">Average Dice score = 0.2027</td> 
<td align="center">Average Dice score = 0.456</td>
</tr>
<tr> 
<td><img src="subtask1 report images/resnetunetBCE-DICELoss_model_predictions.png" width="340"/></td>
<td><img src="subtask1 report images/resnetunet_model_predictions 1.png" width="340"/></td>
</tr> 
</table> 
</div>
<div align="center">
<table>
<tr> 
<td align="center"><b>Focal + BCE + Dice Loss</b></td> 
</tr>
<tr> 
<td align="center">Average IoU score = 0.3613</td> 
</tr>
<tr> 
<td align="center">Average Dice score = 0.5208</td> 
</tr>
<tr> 
<td><img src="subtask1 report images/segformer_model_predictions.png" width="340"/></td>
</tr> 
</table> 
</div>

## <u>Conclusion</u>
In conclusion, first of all, the ResNet-enabled model seems to have performed the best of all models. This can be attributed to the limit placed on the data-samples (1,500 instead of ~6,300). The diminishing return of attended models and transformer models can be explained this way, since they require more training data.

It could have also been that the focal loss's hyperparameters ($\alpha_t$ and $\gamma_t$) may not have been properly tuned, which resulted in the largely-empty training masks skewing the models' understanding.

In the experiment for various loss functions (and combinations), using Dice-Loss on the ResNet-enabled model seems to have yielded the best results, with Focal+Dice+BCE a close runner-up quantitatively. But qualitatively, Dice-Loss produced the most clean masks of them all, since it directly prioritizes maximizing intersections and minimizing unions.
