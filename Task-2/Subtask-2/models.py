from diffusers import DDPMPipeline,  DDIMScheduler, DDPMScheduler, UNet2DModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import glob
from typing import Literal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Dataset implementation
class ImageSet(Dataset):
    def __init__(self, data_type): #, augment=False
        super().__init__()
        preproDone = not (glob.glob(f"./Pre-processed/X_{data_type}.npy")==[])
        assert preproDone, "Please run the pre-processing script first! Or check dataset type passed."
        self.X = np.load(f"./Pre-processed/X_{data_type}.npy", mmap_mode='r')
        self.y = np.load(f"./Pre-processed/y_{data_type}.npy", mmap_mode='r')
        # self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array(self.X[idx])), int(self.y[idx]))



# DDMP Model implementation, in the form of a Torch wrapper around HuggingFace diffuser pipeline
class DDPM_Model(nn.Module):
    def __init__(self, num_train_steps, scheduler:Literal['ddpm','ddim']='ddpm', beta_schedule:Literal['linear', 'scaled_linear', 'squaredcos_cap_v2', 'sigmoid']='linear', dropout=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_pipe = DDPMPipeline.from_pretrained('ddpm-pretrained-butterflies-john').to(device)
        if dropout:
            config = self.image_pipe.unet.config
            config.dropout = dropout
            new_unet = UNet2DModel(**config).to(device)
            old_state_dict = self.image_pipe.unet.state_dict()
            new_unet.load_state_dict(old_state_dict, strict=False)
            self.image_pipe.unet = new_unet
        if scheduler=='ddpm':
            self.image_pipe.scheduler = DDPMScheduler(num_train_steps, beta_schedule=beta_schedule)
        elif scheduler=='ddim':
            self.image_pipe.scheduler = DDIMScheduler(num_train_steps, beta_schedule=beta_schedule)
        else:
            raise Exception("Invalid Scheduler Class")


    def forward(self, noisy_image, timesteps):
        images = self.image_pipe.unet(noisy_image, timesteps).images
        return images[0]

