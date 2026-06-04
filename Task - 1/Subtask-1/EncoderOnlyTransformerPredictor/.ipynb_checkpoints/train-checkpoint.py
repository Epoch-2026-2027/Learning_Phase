from torch.utils.data import DataLoader
from model import *
from dataset import createDatasets
import pandas as pd
dataset = pd.read_csv("../ranking_dataset.csv") 

(train_dataset, test_dataset, valid_dataset),(train_size, test_size, valid_size) = createDatasets(dataset,.1,.1)

train_data = DataLoader(train_dataset,batch_size=64)
valid_data = DataLoader(valid_dataset,batch_size=4096)
test_data = DataLoader(test_dataset,batch_size=4096)

max_epochs = 100
learning_rate = 1e-3

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cuda"
model = EncoderOnlyTransformerPredictor(device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.AdamW(model.parameters(),lr = learning_rate)

model.train()
for i in range(max_epochs):
    s = 0
    for (x,y) in train_data:
        y = y.to(device)
        x = x.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        loss.backward()
        s+=loss.item()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        total = 0
        for (x,y) in valid_data:
            y = y.to(device)
            x = x.to(device)
            pred = model(x)
            pred = pred.reshape(-1,10,10)
            pred = torch.argmax(pred,dim=-1).reshape(-1)
            y = y.reshape(-1,10,10)
            y = torch.argmax(y,dim=-1).reshape(-1)
            total += (y == pred).sum().cpu() 
        if (i%10 == 0):
            print(total/valid_size/10)

torch.save(model.state_dict(), f"./model_parameter.trhmodel")


model.eval()

total = 0
total2 = 0
with torch.no_grad():
    for (x,y) in test_data:
        y = y.to(device)
        x = x.to(device)
        pred = model(x)
        pred = pred.reshape(-1,10,10)
        pred = torch.argmax(pred,dim=-1).reshape(-1)
        y = y.reshape(-1,10,10)
        y = torch.argmax(y,dim=-1).reshape(-1)
        z = (y == pred)
        total2+=z.sum()
        z = z.reshape(-1,10)
        total += torch.where(torch.sum(z,dim = -1) == 10, 1,0).sum()

print(total2/test_size/10,total/test_size)
