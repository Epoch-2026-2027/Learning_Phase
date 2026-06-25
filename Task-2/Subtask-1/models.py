import torch, torch.nn as nn
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, AutoConfig
import numpy as np, pandas as pd
import glob
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.io import read_image
import torchvision.transforms.functional as F
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = [lambda x:x, lambda x: F.rotate(x,90), lambda x: F.rotate(x,180), lambda x: F.rotate(x,270),
              lambda x: F.vflip(x), lambda x: F.vflip(F.rotate(x,90)), lambda x: F.vflip(F.rotate(x,180)), lambda x: F.vflip(F.rotate(x,270))]


# Dataset implementation
class ImageSet(Dataset):
    def __init__(self, data_type, augment=False):
        super().__init__()
        preproDone = not (glob.glob(f"./Preprocessed_Data/X_{data_type}.npy")==[])
        assert preproDone, "Please run the pre-processing script first! Or check dataset type passed."
        self.X = np.load(f"./Preprocessed_Data/X_{data_type}.npy", mmap_mode='r')
        self.y = np.load(f"./Preprocessed_Data/y_{data_type}.npy", mmap_mode='r')
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.augment:
            tran = random.choice(transforms)
            return (tran(torch.from_numpy(np.array(self.X[idx]))), tran(torch.from_numpy(np.array(self.y[idx])).to(dtype=torch.float)))
        else:
            return (torch.from_numpy(np.array(self.X[idx])), torch.from_numpy(np.array(self.y[idx])).to(dtype=torch.float))




# Defining some commonly recurring architecture elements as modules
class DoubleConv(nn.Module):
    def __init__(self, inp_channel, out_channel, kernel_size=3, stride=1, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(inp_channel, out_channel, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class UpConv(nn.Module):
    def __init__(self, inp_channel, out_channel, kernel_size=2, stride=2, padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (out_channel!=inp_channel or inp_channel//out_channel!=2), "Invalid use of input and output channels!"
        self.upconv = nn.ConvTranspose2d(inp_channel, out_channel, kernel_size, stride, padding)
        self.doubleconv = DoubleConv(out_channel*2, out_channel)

    def forward(self, down, res):
        up = self.upconv(down)
        # print( res.shape, up.shape, down.shape)
        x = torch.concatenate([res,up], dim=1)
        # print(x.shape)
        x = self.doubleconv(x)
        return x


# Classic U-Net implementation
class ClassicUNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder initialisations, Input - Bx3x256x256
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dec1 = DoubleConv(3, 64) # Double Encoder Convolution layer 1   # Output after maxpooling-> Bx64x128x128
        self.dec2 = DoubleConv(64, 128)    # Output after maxpooling-> Bx128x64x64
        self.dec3 = DoubleConv(128, 256)   # Output after maxpooling-> Bx256x32x32
        self.dec4 = DoubleConv(256, 512)   # Output after maxpooling-> Bx512x16x16
        self.dec5 = DoubleConv(512, 1024)  # Output after maxpooling-> Bx1024x8x8

        # Decoder initialisations, Input - dec5 output - Bx1024x8x8
        self.duc1 = UpConv(1024,512) # Decoder Up-Convolution layer 1   # Output-> Bx512x16x16
        self.duc2 = UpConv(512,256)   # Output-> Bx256x32x32
        self.duc3 = UpConv(256,128)   # Output-> Bx128x64x64
        self.duc4 = UpConv(128,64)    # Output-> Bx64x512x512

        self.fconv = nn.Conv2d(64, 1, 1)
        self.sigm = nn.Sigmoid()


    def forward(self, input):
        # Encoder
        # print(input.shape)
        x1 = self.dec1(input)
        x2 = self.maxpool(x1)
        x2 = self.dec2(x2)
        x3 = self.maxpool(x2)
        x3 = self.dec3(x3)
        x4 = self.maxpool(x3)
        x4 = self.dec4(x4)
        x = self.maxpool(x4)
        x = self.dec5(x)
        
        # Decoder
        x = self.duc1(x, x4)
        x = self.duc2(x, x3)
        x = self.duc3(x, x2)
        x = self.duc4(x, x1)

        # Final
        x = self.fconv(x)
        # x = self.sigm(x)
        return x


# Attention U-Net Implementation
class AttentionGate(nn.Module):
    def __init__(self, xfilters, gfilters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xconv = nn.Conv2d(xfilters, xfilters, kernel_size=1, stride=2)
        self.gconv = nn.Conv2d(gfilters, xfilters, kernel_size=1)
        self.relu = nn.ReLU()
        self.resconv = nn.Conv2d(xfilters, 1, kernel_size=1)
        self.sigm = nn.Sigmoid()

    def forward(self, x, g):
        x_ = self.xconv(x)
        g = self.gconv(g)
        res = x_ + g
        res = self.relu(res)
        res = self.resconv(res)
        res = self.sigm(res)
        res = torch.nn.functional.interpolate(res, scale_factor=2, mode='bilinear', align_corners=False)
        out = x * res
        return out

class AttentionUNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder initialisations, Input - Bx3x1024x1024
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dec1 = DoubleConv(3, 64) # Double Encoder Convolution layer 1   # Output after maxpooling-> Bx64x512x512
        self.dec2 = DoubleConv(64, 128)    # Output after maxpooling-> Bx128x256x256
        self.dec3 = DoubleConv(128, 256)   # Output after maxpooling-> Bx256x128x128
        self.dec4 = DoubleConv(256, 512)   # Output after maxpooling-> Bx512x64x64
        self.dec5 = DoubleConv(512, 1024)  # Output after maxpooling-> Bx1024x64x64

        # Decoder initialisations, Input - dec5 output - Bx1024x64x64
        self.ag1 = AttentionGate(512,1024)
        self.duc1 = UpConv(1024,512) # Decoder Up-Convolution layer 1   # Output-> Bx512x128x128
        self.ag2 = AttentionGate(256,512)
        self.duc2 = UpConv(512,256)   # Output-> Bx256x128x128
        self.ag3 = AttentionGate(128,256)
        self.duc3 = UpConv(256,128)   # Output-> Bx128x256x256
        self.ag4 = AttentionGate(64,128)
        self.duc4 = UpConv(128,64)    # Output-> Bx64x512x512

        self.fconv = nn.Conv2d(64, 1, 1)


    def forward(self, input):
        # Encoder
        # print(input.shape)
        x1 = self.dec1(input)
        x2 = self.maxpool(x1)
        x2 = self.dec2(x2)
        x3 = self.maxpool(x2)
        x3 = self.dec3(x3)
        x4 = self.maxpool(x3)
        x4 = self.dec4(x4)
        x = self.maxpool(x4)
        x = self.dec5(x)
        
        # Decoder
        x = self.duc1(x, self.ag1(x4, x))
        x = self.duc2(x, self.ag2(x3, x))
        x = self.duc3(x, self.ag3(x2, x))
        x = self.duc4(x, self.ag4(x1, x))

        # Final
        x = self.fconv(x)
        return x


# Pre-trained backbone with U-Net : Resnet(18) U-net
class ResNetUNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder initialisations, ResNet18 backbone
        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        return_nodes = {
            'relu': 'stem',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4'
        }
        self.resnet = create_feature_extractor(resnet, return_nodes=return_nodes)

        # Decoder initialisations, Input - backbone output with residual connections
        self.duc1 = UpConv(512,256) # Decoder Up-Convolution layer 1  
        self.duc2 = UpConv(256,128)   
        self.duc3 = UpConv(128,64)  
        self.duc4 = UpConv(64,64)   
        self.duc5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fconv = nn.Conv2d(64, 1, 1)


    def forward(self, input):
        # Encoder
        # print(input.shape)
        f_maps = self.resnet(input)
        x1 = f_maps['stem']
        x2 = f_maps['layer1']
        x3 = f_maps['layer2']
        x4 = f_maps['layer3']
        x = f_maps['layer4']
        
        # Decoder
        x = self.duc1(x, x4)
        x = self.duc2(x, x3)
        x = self.duc3(x, x2)
        x = self.duc4(x, x1)
        x = self.relu(self.duc5(x))

        # Final
        x = self.fconv(x)
        return x



# Pre-trained backbone with U-Net : Resnet(18) U-net, with Attention Gates
class ResNetAttendedUNet(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder initialisations, ResNet18 backbone
        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        return_nodes = {
            'relu': 'stem',
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4'
        }
        self.resnet = create_feature_extractor(resnet, return_nodes=return_nodes)

        # Decoder initialisations, Input - backbone output with residual connections
        self.ag1 = AttentionGate(256,512)
        self.duc1 = UpConv(512,256) # Decoder Up-Convolution layer 1
        self.ag2 = AttentionGate(128,256)  
        self.duc2 = UpConv(256,128)   
        self.ag3 = AttentionGate(64,128)
        self.duc3 = UpConv(128,64)  
        self.ag4 = AttentionGate(64,64)
        self.duc4 = UpConv(64,64)   
        self.duc5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.fconv = nn.Conv2d(64, 1, 1)


    def forward(self, input):
        # Encoder
        # print(input.shape)
        f_maps = self.resnet(input)
        x1 = f_maps['stem']
        x2 = f_maps['layer1']
        x3 = f_maps['layer2']
        x4 = f_maps['layer3']
        x = f_maps['layer4']
        
        # Decoder
        x = self.duc1(x, self.ag1(x4, x))
        x = self.duc2(x, self.ag2(x3, x))
        x = self.duc3(x, self.ag3(x2, x))
        x = self.duc4(x, self.ag4(x1, x))
        x = self.relu(self.duc5(x))

        # Final
        x = self.fconv(x)
        return x



# Transformer based Segmentation model
# SegFormer implementation (Wrapper around huggingface pre-trained model, to be fine-tuned)
class SegFormerWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1, ignore_mismatched_sizes=True
            )
        else:
            self.model = SegformerForSemanticSegmentation(AutoConfig.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1, ignore_mismatched_sizes=True))
            
    def forward(self, x):
        out = self.model(pixel_values=x).logits
        return torch.nn.functional.interpolate(out, size=(256,256), mode='bilinear', align_corners=False)