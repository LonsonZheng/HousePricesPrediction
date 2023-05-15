import torch
import torch.nn as nn
import torch.nn.functional  as F
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import random
import time
import os
from tensorboardX import SummaryWriter
class Linear_Net(nn.Module):
    def __init__(self,in_dim=14,out_dim=1,hidden=100) :
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
        return self.x.shape[0]
    def __getitem__(self, index) :
        
        inps=torch.FloatTensor(np.array(self.x.iloc[index]))
        labels=torch.FloatTensor(np.array(self.y.iloc[index]))
        return inps,labels

class PriceData_Test(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        assert x.shape[0]==y.shape[0]
        self.x=x
        self.y=y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index) :
        
        inps=torch.FloatTensor(np.array(self.x.iloc[index]))
        labels=torch.FloatTensor(np.array(self.y.iloc[index]))
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

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def save_checkpoint(model_path, epoch ,model, optimizer,flag=None):
    state = {
        'epoch': epoch,

        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if flag is not None:
        torch.save(state, os.path.join(model_path, 'best.pth' ))
    else:
        torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))
    return 
if __name__=='__main__':
    current_time = time.strftime("%Y-%m-%dT%H-%M", time.localtime())
    visual_dir=os.path.join('visual',current_time)
    save_dir=os.path.join('exp',current_time)
    os.makedirs(save_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=visual_dir)

    random.seed(816)
    losses = AverageMeter()
    epoch=100
    model=Linear_Net().cuda()
    data_root="data_pre3.csv"
    data = pd.read_csv(data_root, encoding="gbk")
    # for col in data.columns:
    #     print(col+' : '+ str(data[col].dtype))
    # process_col=[ "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor", "elevator", "subway", "district", "tradeYear"]
    data.drop(["url", "id", "price","constructionTime"],axis=1, inplace=True)
    
    process_col=["DOM", "followers", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor", "renovationCondition", "buildingStructure", "elevator", "subway", "district", "tradeyear"]
    #data=data[process_col]
    data = data.sample(frac=1)
    #data=data.div(data.max())

    #x = data.copy()
    x =data[process_col]
    for nkey in process_col:
        x[nkey]=x[nkey].div(x[nkey].max())
    #x.drop(['totalPrice'],axis=1, inplace=True)
    
    y = data['totalPrice']
    scale=y.max()
    y=y.div(y.max())
    spilt_point=int(0.8*data.shape[0])
    x_train = x.iloc[0:spilt_point,:]
    x_test = x.iloc[spilt_point:,:]
    y_train = y.iloc[0:spilt_point]
    y_test = y.iloc[spilt_point:]

    train_dataset=PriceData_Train(x_train,y_train)
    test_dataset=PriceData_Test(x_test,y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True,pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, pin_memory=True)

    optimizer = optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999),weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=1e-6)
    criterion_rmse = Loss_RMSE()
    iter_per_epoch=len(train_loader)
    test_loss_min=1e9
    for j in range(epoch):
        for i, (inps, labels) in enumerate(train_loader):
            inps=inps.cuda()
            labels=labels.cuda()
            model.train()
            optimizer.zero_grad()
            out=model(inps)
            loss=criterion_rmse(out,labels)
            losses.update(loss)
            writer.add_scalar('loss/train', loss,j*iter_per_epoch+i)
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if i % 20 == 0:
                print('[epoch:%d, iter:%d], lr=%.9f, train_loss.avg=%.9f, train_loss=%.9f'% (j,i,lr,losses.avg, loss.data))
        test_loss=0
        num=0
        print('testing...')
        for i, (inps, labels) in enumerate(test_loader):
            inps=inps.cuda()
            labels=labels.cuda()
            model.eval()
            with torch.no_grad():
                out=model(inps)
                test_loss_now=criterion_rmse(out,labels)
                if i % 20 == 0:
                    print('[test_iter:%d] ,test_loss=%.9f'% (i, test_loss_now.data))
                test_loss+=test_loss_now
            num+=1
        test_loss=test_loss.data/num
        if test_loss < test_loss_min:
            test_loss_min=test_loss
            save_checkpoint(save_dir,j,model,optimizer,1)
        print('[test_epoch:%d] ,test_loss=%.9f'% (j, test_loss))   
        writer.add_scalar('loss/test', test_loss,j)
        scheduler.step()
    writer.close()
        
