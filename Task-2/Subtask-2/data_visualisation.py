import torch
import torchvision
import torchvision.transforms as T
import numpy as np 
import PIL
import pandas as pd
import io
import glob
import matplotlib.pyplot as plt

# Filtering out and keeping only airplane (classid=0), cat (classid=3), dog (classid=5), ship (classid=8)

classes = [0,3,5,8]

_train_set = pd.read_parquet('./CIFAR-10 Datasets/cifar10_train_set.parquet')
_train_set = _train_set[_train_set['label'].isin(classes)].to_numpy()
test_set = pd.read_parquet('./CIFAR-10 Datasets/cifar10_test_set.parquet')
test_set = test_set[test_set['label'].isin(classes)].to_numpy()

dataset = np.concat([_train_set, test_set], axis=0)

if not glob.glob('./Pre-processed/norm_metrics.npz'):
    means = []
    stds = []
    for label in classes:
        mean, m2 = torch.zeros((3,)), torch.zeros((3,))
        for ind, (img_bytes, cla) in enumerate(dataset, start=1):
            if cla==label:
                img_tensor = PIL.Image.open(io.BytesIO(img_bytes['bytes']))
                img_tensor = T.PILToTensor()(img_tensor).to(dtype=torch.float)
                d1 = torch.mean(img_tensor, dim=(1,2)) - mean
                mean = mean + d1/ind
                d2 = torch.mean(img_tensor, dim=(1,2)) - mean
                m2 = m2 + d1*d2
        std = torch.sqrt(m2/ind)
        means.append(mean.numpy())
        stds.append(std.numpy())
    np.savez('./Pre-processed/data_metrics.npz', means=np.array(means), stds=np.array(stds))
else:
    dat = np.load('./Pre-processed/data_metrics.npz')
    means = dat['means']
    stds = dat['stds']


for mean,std,label in zip(means, stds, classes):
    print("Class ID:",label)
    print("Mean:", np.round(mean, 4))
    print("Standard Dev:",np.round(std, 4))
    print()


for label in classes:
    imgs = []
    i=0
    for ind, (img_bytes, cla) in enumerate(dataset, start=1):
        if cla==label:
            if i%400==0:
                img_tensor = PIL.Image.open(io.BytesIO(img_bytes['bytes']))
                img_tensor = T.PILToTensor()(img_tensor).to(dtype=torch.float)
                imgs.append(img_tensor)
            i+=1

    grid = torchvision.utils.make_grid(imgs, padding=25, nrow=3, normalize=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.title(f"CIFAR-10 Examples for Class ID={label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./Pre-processed/samples_{label}", dpi=150)
    plt.close()