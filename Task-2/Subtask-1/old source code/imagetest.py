import torch, torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import PIL
from matplotlib import pyplot as plt
from models import device, ImageSet, ClassicUNet, AttentionUNet, ResNetUNet
import glob


print("ping!!!")

img_tensor = PIL.Image.open("CV_Data/sats/104_sat.jpg").resize((512, 512), PIL.Image.BILINEAR)
img_tensor = T.PILToTensor()(img_tensor).float().unsqueeze(dim=0).to(device)

# print(imgtensor.shape) # B, C, H, W

model = ResNetUNet(num_classes=2)
model.to(device)

res = model.forward(img_tensor)
print(res, res.shape)

# arr = imgtensor.numpy()[0]
# arr = np.squeeze(arr)

# plt.imshow(arr, cmap="cool_r")
# plt.show()

# mask_image_paths = sorted(glob.glob("./CV_Data/masks/*.png"))
# sat_image_paths = sorted(glob.glob("./CV_Data/sats/*.jpg"))

# masks = []
# sats = []

# # 2. Loop, read directly as tensor, resize, and append
# for ind,(path1,path2) in enumerate(zip(mask_image_paths, sat_image_paths)):
#     img_tensor1 = read_image(path1)
#     img_tensor1, _ = torch.max(img_tensor1, dim=0, keepdim=True)
#     img_tensor1 = (img_tensor1 > 0).to(torch.int)
#     img_tensor2 = read_image(path2) 
#     # img_tensor = resize_transform(img_tensor)
#     masks.append(img_tensor1)
#     sats.append(img_tensor2)
#     if ind==100:
#         break

# batched_masks = torch.stack(masks).to(torch.int)
# batched_sats = torch.stack(sats).to(torch.int)
# # print(batched_masks.shape)
# # print(batched_sats.shape)

# fig=plt.figure()
# for i in range(101):
#     plt.subplot(1,2,1)
#     plt.imshow(batched_masks[i][0])
#     plt.subplot(1,2,2)
#     plt.imshow(torch.permute(batched_sats[i], (1,2,0)).numpy())
#     plt.show()