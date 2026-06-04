test_data = DataLoader(test_dataset, batch_size=128)

model.eval()
total = 0
total2 = 0
n = 0
with torch.no_grad():
    for (x,y) in test_data:
        y = y.to(device)
        x = x.to(device)
        pred = model(x)
        pred = pred.reshape(-1,10,10)
        pred = torch.argmax(pred,axis=1).reshape(-1)
        y = y.reshape(-1,10,10)
        y = torch.argmax(y,axis=1).reshape(-1)
        z = (y == pred)
        total2+=z.sum()
        z = z.reshape(-1,10)
        total += torch.where(torch.sum(z,dim = 1) == 10, 1,0).sum()

print(total2/test_size/10,total/test_size)