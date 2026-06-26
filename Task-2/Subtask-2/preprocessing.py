import torch
import torchvision.transforms as T 
from sklearn.model_selection import train_test_split
import numpy as np
from npy_append_array import NpyAppendArray
import pandas as pd
import io
import PIL
import glob

print("Libraries imported!")


# Filtering out and keeping only airplane (classid=0), cat (classid=3), dog (classid=5), ship (classid=8)
_train_set = pd.read_parquet('./CIFAR-10 Datasets/cifar10_train_set.parquet')
_train_set = _train_set[_train_set['label'].isin([0,3,5,8])].to_numpy()

train_set, val_set = train_test_split(_train_set, test_size=0.2, random_state=42, shuffle=True)

test_set = pd.read_parquet('./CIFAR-10 Datasets/cifar10_test_set.parquet')
test_set = test_set[test_set['label'].isin([0,3,5,8])].to_numpy()

# Pre-processing composition
preprocess = T.Compose(
    [
        T.ToTensor(), 
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),  
    ]
)

# Writing a image-label pair to an npy file
def write_pairs(set, tag):
    with NpyAppendArray(f'Pre-processed/X_{tag}.npy') as fileX, NpyAppendArray(f'Pre-processed/y_{tag}.npy') as filey:
        for ind, (img_bytes, label) in enumerate(set):
            img_tensor = PIL.Image.open(io.BytesIO(img_bytes['bytes']))
            img_tensor = preprocess(img_tensor)
            # Writing Image-Label pair
            fileX.append(np.array([img_tensor.numpy()]))
            filey.append(np.array([label]))


print("Initiating pre-processing of image files...")

write_pairs(train_set, 'train')
write_pairs(val_set, 'val')
write_pairs(test_set, 'test')