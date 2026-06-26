# Methodology:
The road segmentation task has been explored on the Massachusetts Roads Dataset, using
1. A UNet (A Baseline Model)

Training and Validation Loss Plots for the UNet:

![Training and Validation Loss Plots for the UNet](Images/LearningPhase_Task2_Subtask1_UNet_Loss.png)

2. An Attention UNet

Training and Validation Loss Plots for the Attenion UNet:

![Training and Validation Loss Plots for the Attention UNet](Images/LearningPhase_Task2_Subtask1_Attention_UNet_Loss.png)

Both the models were trained for 100 epochs with a learning rate of 1e-4.

For both the models, the images have first been converted to their RGB values for the input set and grayscale values for the label mask, and then downscaled from their original resolution of 1500 x 1500 to that of 256 x 256. The framework for this is from the PIL library. The reason for the downscaling of the images was in order to consume less RAM during training and inference and to make the model architecture more compact.

All the models have been trained using the PyTorch framework with the Adam optimizer, and using the Dice Loss. The Dice Loss function from the Segmentation Models Pytorch library was used for implementing the same. After training and validating the models, their training and validation losses have been plotted. The evaluation of the models has been done on the basis of the following two metrics:
1. IoU
   
2. Dice Coefficient

# Architectural Choices:
## UNet
A heavy class imbalance was seen in the road vs non-road classes in the label masks for the segmentation throughout the dataset. When evaluated over a batch of the first 100 images in the training set, it was seen that the net percentage of the road vs non-road class was 4.89 % vs 95.11 % respectively. In order to effectively handle the training of the model over this highly imbalanced data, Dice Loss was chosen as the criterion. Where the original UNet architecture was designed to take images with a resolution of 572 x 572 as input, with 5 blocks each in both the contracting and expanding path, since the resolution in our case is much smaller (256 x 256), only 4 blocks were implemented in this model for both paths.
## Attention UNet
The reason behind chosing to compare the baseline to an Attention UNet was to see how well the addition of the additive attention gate component translates towards improving the quality of the model for this particular task of binary image segmentation in a highly imbalanced dataset. Similar to the baseline model, this model was also trained using Dice Loss and had 4 blocks in both the contracting and expanding paths.

# Evaluation Metrics:
| Model | Test IoU | Test Dice Coefficient |
| :--- | :--- | :--- |
| UNet | 0.5490 | 0.7088 |
| Attention UNet | 0.5430 | 0.7038 |

# Qualitative Outputs:
The following is a sample imag from the test dataset and its corresponding ground-truth binary segmentation mask, compared with the model's output for the segmentation mask:
## 1. UNet

![Sample_Image_with_Prediction for the UNet](Images/LearningPhase_Task2_Subtask1_UNet_Sample_Prediction.png)

Though the model understands the basic high-level structure of the code and follows the expected output correctly for the first few tokens, it can clearly be seen struggling to correctly identify the positions of opening and closing parantheses, as well as the right method names and variable names.

## 2. Attention UNet

![Sample_Image_with_Prediction for the Attention_UNet](Images/LearningPhase_Task2_Subtask1_Attention_UNet_Sample_Prediction.png)

A much better match is found here between the expected and generated outputs as compared to the simple LSTM. The model understands the high-level code layout as well as the parantheses placement very well. However, it confuses the name of VAR_1 and VAR_2 mid-way through the code, indicating that there is still a possibilty of the model being confused between the exact method names and variable names, especially when it is near the middle of the expected output. The encoder being a bidirectional LSTM clearly helps it get the begining and ending parts of the fixed code sequence right, but as mentioned, it may struggle with the detailed information in the middle of the sequence.
