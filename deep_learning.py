import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import random
from tensorboardX import SummaryWriter
class Linear_Net(nn.Module):
    def __init__(self,in_dim=23,out_dim=1,hidden=1000) :
        super().__init__()
        self.body=nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,out_dim),
            
        )

    def forward(self,inp):
        return  self.body(inp)
    
class PriceData_Train(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        assert x.shape[0]==y.shape[0]
        self.x=x
        self.y=y
    def __len__(self):
        return x.shape[0]
    def __getitem__(self, index) :
        
        inps=torch.FloatTensor(np.array(x.iloc[index]))
        labels=torch.FloatTensor(np.array(y.iloc[index]))
        return inps,labels

class PriceData_Test(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        assert x.shape[0]==y.shape[0]
        self.x=x
        self.y=y
    def __len__(self):
        return x.shape[0]
    def __getitem__(self, index) :
        
        inps=torch.FloatTensor(np.array(x.iloc[index]))
        labels=torch.FloatTensor(np.array(y.iloc[index]))
        return inps,labels
class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        outputs=outputs.squeeze(-1)
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.reshape(-1)))
        return rmse    
if __name__=='__main__':
    writer = SummaryWriter(log_dir='visual\\')
    random.seed(816)

    epoch=100
    model=Linear_Net()
    data_root="data_pre3.csv"
    data = pd.read_csv(data_root, encoding="gbk")
    # for col in data.columns:
    #     print(col+' : '+ str(data[col].dtype))
    data.drop(["url", "id", "price","constructionTime"],axis=1, inplace=True)
    data = data.sample(frac=1)
    data=data.div(data.max())

    x = data.copy()
    x.drop(['totalPrice'],axis=1, inplace=True)
    
    y = data['totalPrice']
    
    x_train = data.iloc[0:int(0.8*data.shape[0]),:]
    x_test = data.iloc[int(0.8*data.shape[0]):,:]
    y_train = data.iloc[0:int(0.8*data.shape[0])]
    y_test = data.iloc[int(0.8*data.shape[0]):]

    train_dataset=PriceData_Train(x_train,y_train)
    test_dataset=PriceData_Test(x_test,y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True,pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    optimizer = optimizer = optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=1e-6)
    criterion_rmse = Loss_RMSE()
    
    for j in range(epoch):
        for i, (inps, labels) in enumerate(train_loader):
            
            model.train()
            optimizer.zero_grad()
            out=model(inps)
            loss=criterion_rmse(out,labels)
            #writer.add_scalar('loss/train', loss,j+i)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print('[iter:%d] ,train_loss=%.9f'% (i, loss.data))
        test_loss=0
        num=0
        for i, (inps, labels) in enumerate(test_loader):
            model.eval()

            with torch.no_grad():
                out=model(inps)
                test_loss+=criterion_rmse(out,labels)
            num+=1
        print('[epoch:%d] ,test_loss=%.9f'% (j, test_loss.data/num))   
        scheduler.step()
        
