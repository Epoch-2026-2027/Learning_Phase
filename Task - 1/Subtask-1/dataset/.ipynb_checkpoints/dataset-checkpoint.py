from torch.utils.data import Dataset,DataLoader,random_split

class NumArrayDataset(Dataset):
    def __init__(self,data):
        self.data = np.array(data,dtype = "int")
        self.x = torch.Tensor(self.data[:,:10])
        self.y = self.data[:,10:]
        tmp = np.zeros((len(self.data),10,10))
        for i in range(len(self.data)):
            for j in range(10):
                tmp[i,j,self.y[i,j]] = 1
        self.y = tmp.reshape(-1,100)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,i):
        self.x[i] -= self.x[i].mean()
        self.x[i] /= self.x[i].std()
        return (self.x[i],self.y[i])
        
        

valid_ratio = .1
test_ratio = .1

test_size = int(test_ratio*len(dataset_trh))
valid_size = int(valid_ratio*len(dataset_trh))
training_size = len(dataset_trh) - test_size - valid_size


(train_dataset, test_dataset, valid_dataset) = random_split(dataset_trh, [training_size,test_size,valid_size])

train_data = DataLoader(train_dataset, batch_size=128)
valid_data = DataLoader(valid_dataset, batch_size=128)

def createDatasets(valid_ratio = .1, test_ratio = .1):
    
    test_size = int(test_ratio*len(dataset_trh))
    valid_size = int(valid_ratio*len(dataset_trh))
    training_size = len(dataset_trh) - test_size - valid_size
    
    (train_dataset, test_dataset, valid_dataset) = random_split(dataset_trh, [training_size,test_size,valid_size])

    return (train_dataset, test_dataset, valid_dataset), (train_size, test_size, valid_size) 
    