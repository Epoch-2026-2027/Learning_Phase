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
learning_rate = 2e-3

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = SimpleRNNPredictor(128,device)
model.to(device)
loss_fn = nn.CrossEntropyLoss()

opt = torch.optim.AdamW(model.parameters(),lr = learning_rate)
accuracy = []
losses = []
model.train()
for i in range(max_epochs):
    s = 0
    for (x,y) in train_data:
        y = y.to(device)
        x = x.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        losses.append(loss.item())
        loss.backward()
        s+=loss.item()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        total = 0
        for (x,y) in valid_data:
            y = y.to(device)
            x = x.to(device)
            pred = model(x).transpose(-1,-2)
            pred = torch.softmax(pred,dim=-1)
            pred = torch.argmax(pred,dim=-1)
            total += (y == pred).reshape(-1).sum().cpu().item() 
            accuracy.append(total/valid_size/10) 
        if (i%10 == 0):
            print(total/valid_size/10)

torch.save(model.state_dict(), f"./model_parameter.trhmodel")
with open("log.txt","w") as f:
    f.write("\n".join(list(map(str,accuracy))))
with open("log2.txt","w") as f:
    f.write("\n".join(list(map(str,losses))))
model.eval()

total = 0
total2 = 0
total3 = 0
total4 = 0
avg =0
with torch.no_grad():
    for (x,y) in test_data:
        y = y.to(device)
        x = x.to(device)
        pred = model(x).transpose(-1,-2)
        pred = torch.softmax(pred,dim=-1)
        pred = torch.argmax(pred,dim=-1)
        z = (y == pred).reshape(-1)
        total+=z.sum()
        z = z.reshape(-1,10)
        total2 += torch.where(torch.sum(z,dim = -1) == 10, 1,0).sum()
        total3 += torch.where(abs(torch.sum(z,dim = -1)-10)<=1, 1,0).sum()
        total4 += torch.where(abs(torch.sum(z,dim = -1)-10)<=2, 1,0).sum()
        avg += abs(torch.sum(z,dim = -1)-10).sum()

print("Test Accuracy (%):",total/test_size/10*100)
print("Full sequence correct (%):",total2/test_size*100)
print("Atmost 1 incorect sequence correct (%):",total3/test_size*100)
print("Atmost 2 incorect sequence correct (%):",total4/test_size*100)
print("Average no of rank predicted incorect in a sequence :",avg/test_size)

with torch.no_grad():
    for i in range(4,50,5):
        (x,y) = test_dataset[i]
        x = x.to(device).reshape(1,10)
        pred = model(x).transpose(-1,-2)
        pred = pred.reshape(10,10).cpu()
        pred = torch.argmax(pred,dim=-1).reshape(-1)
        print(f"{i+1} testcase")
        print("x : ",x.cpu().numpy())
        print("pred : ",pred.numpy())
        print("true : ",y)
