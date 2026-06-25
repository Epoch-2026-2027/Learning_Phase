# data.py

from pathlib import Path
import pickle
import numpy as np
import torch


RAW_DIR = Path(r"C:\Users\tilak\OneDrive\Documents\Tilak Asodariya\AI ML\Epoch\CORES\Task-2\SubTask-2\cifar-10-batches-py")
PROCESSED_DIR = Path("processed")

PROCESSED_DIR.mkdir(exist_ok=True)


def unpickle(file):
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data

def load_batch(batch_path):
    batch = unpickle(batch_path)
    images = batch[b"data"]
    labels = batch[b"labels"]
    images = images.reshape(-1, 3, 32, 32)
    return images, np.array(labels)


train_images = []
train_labels = []

for i in range(1, 6):
    path = RAW_DIR / f"data_batch_{i}"
    imgs, lbls = load_batch(path)
    train_images.append(imgs)
    train_labels.append(lbls)

train_images = np.concatenate(train_images, axis=0)
train_labels = np.concatenate(train_labels, axis=0)



test_images, test_labels = load_batch(RAW_DIR / "test_batch")

train_images = torch.tensor(train_images,dtype=torch.float32)
test_images = torch.tensor(test_images,dtype=torch.float32)

#[0,255]->[0,1]
train_images /= 255.0
test_images /= 255.0

#[0,1]->[-1,1]
train_images = (train_images - 0.5) / 0.5
test_images = (test_images - 0.5) / 0.5


train_labels = torch.tensor(train_labels,dtype=torch.long)
test_labels = torch.tensor(test_labels,dtype=torch.long)

meta = unpickle(RAW_DIR / "batches.meta")
label_names = [name.decode("utf-8")for name in meta[b"label_names"]]

torch.save({"images": train_images,"labels": train_labels,},PROCESSED_DIR / "train.pt",)
torch.save({"images": test_images,"labels": test_labels,},PROCESSED_DIR / "test.pt",)
torch.save({"label_names": label_names},PROCESSED_DIR / "meta.pt",)

print("Saved Successfully")
print(f"Train Images : {train_images.shape}")
print(f"Train Labels : {train_labels.shape}")
print(f"Test Images  : {test_images.shape}")
print(f"Test Labels  : {test_labels.shape}")
print()
print("Classes:")
print(label_names)