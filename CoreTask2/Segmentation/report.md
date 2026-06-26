# Road Segmentation from Satellite Images

Dataset: Massachusetts Roads Dataset - https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset

## 1. Task

Build segmentation models that take a satellite image as input and output a binary pixel-wise mask for roads.

## 2. Data Analysis

The dataset has 1171 aerial images at 1500x1500 pixels, covering urban, suburban and rural regions so road density varies across images.

Road pixels make up a small fraction of each image. The mean road fraction across all splits is roughly 4-7%, which is severe class imbalance - a model predicting all background would get over 90% pixel accuracy for free so accuracy is useless and we use IoU and Dice instead.

| | |
|------|-------------|
|train images | 1108 |
|val images | 14 |
|test images | 49 |
|mean road fraction | ~0.05 |

## 3. Dataset and Augmentation

Images are 1500x1500 and training on full res would be prohibitive at batch size greater than 1, so cropped to 512x512. For training: `RandomCrop`, `HorizontalFlip`, then ImageNet normalisation and `ToTensorV2`. Validation uses `CenterCrop` only. The flip is appropriate here since aerial road orientations have no preferred direction.

Masks are read as grayscale and normalised to [0,1] by dividing by 255, single channel binary output with sigmoid + 0.5 threshold at eval time.

## 4. Loss Functions

The road class imbalance (over 90% background) means vanilla BCE just learns to predict background. BCE treats all pixels equally and gives no special weight to the rare road class. Dice optimises overlap directly $\mathcal{L}_{dice} = 1 - \frac{2\sum p_i t_i + \epsilon}{\sum p_i + \sum t_i + \epsilon}$, naturally handling imbalance by using global overlap ratio. BCE+Dice combines both: BCE stabilises early training while Dice pulls the model toward the global overlap metric.

## 5. Models

### 5.1 U-Net Baseline

U-Net with four encoder stages (64, 128, 256, 512 channels), symmetric decoder with skip connections and ConvTranspose2d upsampling. Each stage is a DoubleConv block (Conv-BN-ReLU x2). Skip connections preserve spatial detail of road pixels which would be lost to pooling otherwise.

The weakness is the encoder trains from scratch on roughly 1100 images which isnt a lot of data for learning good feature detectors from random init - the model has to learn edges, textures and road patterns all from nothing.

### 5.2 ResNet50 Encoder U-Net

Replaces the scratch encoder with pretrained ResNet50 via `tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)`. The encoder stages produce feature maps at 5 scales (64, 256, 512, 1024, 2048 channels).

Even though satellite imagery looks different from ImageNet, low-level edge and texture detectors in early ResNet layers transfer well to detecting road boundaries. The pretrained weights give the model a large headstart on feature extraction.

### 5.3 DeepLabV3 (ResNet50)

Uses torchvision's pretrained DeepLabV3, replacing the final classifier head with a single-channel output. DeepLabV3 uses atrous spatial pyramid pooling (ASPP) to aggregate context at multiple scales without reducing spatial resolution, which should help with thin structures like roads where a standard stride encoder loses too much detail.

## 6. Results


All models trained with BCEDiceLoss, Adam, lr=1e-4, 20 epochs, 512x512 crops.

| Model | IoU | Dice |
|-------|-----|------|
| UNet (scratch) | 0.6002 | 0.7235 |
| ResUNet (pretrained) | 0.5963 | 0.7161 |
| DeepLabV3 (pretrained) | 0.5894 | 0.7148 |