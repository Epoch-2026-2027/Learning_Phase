import numpy as np
import pandas as pd


print("Initialising search...")

train_data = pd.read_parquet('.\\Datasets\\small\\train.parquet').to_numpy()
val_data = pd.read_parquet('.\\Datasets\\small\\validation.parquet').to_numpy()
test_data = pd.read_parquet('.\\Datasets\\small\\test.parquet').to_numpy()

print("Stand by!")

vocab = []
max_length = 0
length=[]

search = [train_data,val_data,test_data]

# train_data = pd.read_parquet('.\\Datasets\\medium\\train.parquet').to_numpy()
# val_data = pd.read_parquet('.\\Datasets\\medium\\validation.parquet').to_numpy()
# test_data = pd.read_parquet('.\\Datasets\\medium\\test.parquet').to_numpy()

# search.extend([train_data,val_data,test_data])

for dataset in search:
    for row in dataset[:1000]:
        length.extend([len(row[1]),len(row[2])])
        if max_length<len(row[1]):
            max_length=len(row[1])
        if max_length<len(row[2]):
            max_length=len(row[2])
        for i in row[1]:
            if i not in vocab:
                vocab.append(i)
        for i in row[2]:
            if i not in vocab:
                vocab.append(i)

print("\nZe results!!\n")

print(vocab, len(vocab), max_length, np.percentile(length, 0.9))