import torch, torchvision.transforms as transforms
# from torchvision.io import read_image
import PIL
import torchvision.transforms as T 
from sklearn.model_selection import train_test_split
import numpy as np
from npy_append_array import NpyAppendArray
from matplotlib import pyplot as plt
import glob
import os


print("Libraries loaded! Initiating pre-processing metrics...")


img_dim = 256 


mask_image_paths = sorted(glob.glob("./CV_Data/masks/*.png"))[:1500] # Indexing to reduce no. of data pairs 
sat_image_paths = sorted(glob.glob("./CV_Data/sats/*.jpg"))[:1500]   # from 6.3k to something manageable


# Retrieving mean and std for Normalisation
if not glob.glob('./Preprocessed_Data/norm_metrics.npz'):
    mean, m2 = torch.zeros((3,)), torch.zeros((3,))
    for ind,(_,path) in enumerate(zip(mask_image_paths, sat_image_paths), start=1):
        img_tensor = PIL.Image.open(path).resize((img_dim, img_dim), PIL.Image.BILINEAR)
        img_tensor = T.PILToTensor()(img_tensor).float()
        d1 = torch.mean(img_tensor, dim=(1,2)) - mean
        mean = mean + d1/ind
        d2 = torch.mean(img_tensor, dim=(1,2)) - mean
        m2 = m2 + d1*d2
    std = torch.sqrt(m2/ind)
    np.savez('./Preprocessed_Data/norm_metrics.npz', mean=mean.numpy(), std=std.numpy())
else:
    dat = np.load('./Preprocessed_Data/norm_metrics.npz')
    mean = torch.tensor(dat['mean'])
    std = torch.tensor(dat['std'])


normalise = T.Normalize(mean=mean, std=std)


def write_pairs(X_path, y_path, tag):
    with NpyAppendArray(f'Preprocessed_Data/X_{tag}.npy') as fileX, NpyAppendArray(f'Preprocessed_Data/y_{tag}.npy') as filey:
        for ind,(path2,path1) in enumerate(zip(X_path, y_path)):
            # if ind==500:
            #     break
            img_tensor1 = PIL.Image.open(path1).resize((img_dim, img_dim), PIL.Image.BILINEAR)
            img_tensor1 = T.PILToTensor()(img_tensor1).float()
            img_tensor1, _ = torch.max(img_tensor1, dim=0, keepdim=True)
            img_tensor1 = (img_tensor1 > 0.5).to(torch.int)
            img_tensor2 = PIL.Image.open(path2).resize((img_dim, img_dim), PIL.Image.BILINEAR)
            img_tensor2 = T.PILToTensor()(img_tensor2).float()
            img_tensor2 = normalise(img_tensor2)
            # Writing satellite images
            fileX.append(img_tensor2.unsqueeze(dim=0).numpy())
            # print(os.path.getsize(f'Preprocessed_Data/X_{tag}.npy')) 
            filey.append(img_tensor1.unsqueeze(dim=0).numpy())


print("Initiating pre-processing of image files...")

X_, X_test, y_, y_test =  train_test_split(sat_image_paths, mask_image_paths, test_size=0.1, shuffle=True, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.1/0.9, shuffle=True, random_state=42)

# Saving training data
write_pairs(X_train, y_train, 'train')

# Saving validation data
write_pairs(X_val, y_val, 'val')

# Saving testing data
write_pairs(X_test, y_test, 'test')


# np.savez("train_data.npz", X=X_train, y=y_train)
# np.savez("val_data.npz", X=X_val, y=y_val)
# np.savez("test_data.npz", X=X_test, y=y_test)


# Below is just for debugging purposes ; Checking validity of corresponding satellite_image:mask_image pairs
# fig=plt.figure()
# for i in range(101):
#     plt.subplot(1,2,1)
#     plt.imshow(batched_masks[i][0])
#     plt.subplot(1,2,2)
#     plt.imshow(torch.permute(batched_sats[i], (1,2,0)).numpy())
#     plt.show()