import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import random
import torch.nn as nn
import random
import time
import model
import read_all
from function import cls
#设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
#定义超参数
setup_seed(4689)
d_model=60;head_num=3;block_num=24
dropout=0.1;learning_rate=1e-4;batch_size=144;embedding_len=10
num_epochs=20000;mask_percent=0.4;valid_epoch=100
device='cuda' if torch.cuda.is_available() else 'cpu'
#读取数据
list_station=["SC02"]
list_track_train=read_all.start(list_station,"train",max_ele=25)
for i in range(34):
    print(len(list_track_train[i]))
list_track_valid=read_all.start(list_station,"valid",max_ele=25)
#创建轨迹数据集
train_dataset=[];valid_dataset=[];train_loader=[];valid_loader=[];train_size=[];train_size0=[];valid_size=[]
for i in range(34):
    train_dataset.append([]);valid_dataset.append([]);train_loader.append([]);valid_loader.append([])
    train_size.append([]);train_size0.append([]);valid_size.append([])
for i in range(34):
    train_size[i]=len(list_track_train[i]);valid_size[i]=len(list_track_valid[i])
    if len(list_track_train[i])==0 or len(list_track_valid[i])==0:continue
    train_data=[];valid_data=[]
    for j in range(train_size[i]):train_data.append(list_track_train[i][j])
    for j in range(valid_size[i]):valid_data.append(list_track_valid[i][j])
    train_dataset[i]=cls.trackset0(train_data)
    train_loader[i]=DataLoader(train_dataset[i],batch_size=batch_size,shuffle=True)
    valid_dataset[i]=cls.trackset0(valid_data)
    valid_loader[i]=DataLoader(valid_dataset[i],batch_size=batch_size,shuffle=True)
#初始化模型、损失函数、优化器
model=model.Model(d_model,head_num,block_num,dropout,d_model//head_num,embedding_len,mask_percent)
criterion=nn.MSELoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
train_size_max=float(max(train_size))
for i in range(34):
    train_size0[i]=float(train_size[i]);train_size0[i]=train_size0[i]/train_size_max
#模型训练
list_train_loss=[];list_valid_loss=[]
def estimate():
    with torch.no_grad():
        model.eval()
        #训练集
        train_loss=[]
        for _ in range(valid_epoch):
            p=random.randint(0,33)
            while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
            inputs_snr,inputs_ele,inputs_fre=next(iter(train_loader[p]))
            inputs0=inputs_snr.clone().detach()
            B,L=inputs_snr.shape
            for i in range(B):
                pp=random.uniform(0,1)
                for j in range(L):
                    if random.uniform(0,1)<mask_percent*pp:
                        inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-0.5,0.5)*inputs_snr[i,j]
                    if random.uniform(0,1)<mask_percent/5:
                        inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-2,2)*inputs_snr[i,j]
            outputs=model.forward(inputs_snr,inputs_ele,inputs_fre)
            train_loss.append(criterion(inputs0,outputs).clone().detach().numpy())
        train_loss=np.array(train_loss)
        #验证集
        valid_loss=[]
        for _ in range(valid_epoch):
            p=random.randint(0,33)
            while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
            inputs_snr,inputs_ele,inputs_fre=next(iter(valid_loader[p]))
            inputs0=inputs_snr.clone().detach()
            B,L=inputs_snr.shape
            for i in range(B):
                pp=random.uniform(0,1)
                for j in range(L):
                    if random.uniform(0,1)<mask_percent*pp:
                        inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-0.5,0.5)*inputs_snr[i,j]
                    if random.uniform(0,1)<mask_percent/5:
                        inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-2,2)*inputs_snr[i,j]
            outputs=model.forward(inputs_snr,inputs_ele,inputs_fre)
            valid_loss.append(criterion(inputs0,outputs).clone().detach().numpy())
        valid_loss=np.array(valid_loss)
        model.train()
        return np.sqrt(np.mean(train_loss)),np.sqrt(np.mean(valid_loss))
w=open("output/Loss1-00.txt","w")
for epoch in range(num_epochs):
    p=random.randint(0,33)
    while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
    inputs_snr,inputs_ele,inputs_fre=next(iter(train_loader[p]))
    #随机遮盖
    inputs0=torch.tensor(inputs_snr,dtype=torch.float32)
    B,L=inputs_snr.shape
    for i in range(B):
        pp=random.uniform(0,1)
        for j in range(L):
            if random.uniform(0,1)<mask_percent*pp:inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-0.5,0.5)*inputs_snr[i,j]
            if random.uniform(0,1)<mask_percent/5:inputs_snr[i,j]=inputs_snr[i,j]+random.uniform(-2,2)*inputs_snr[i,j]
    outputs=model.forward(inputs_snr,inputs_ele,inputs_fre)
    loss=criterion(inputs0,outputs)
    optimizer.zero_grad();loss.backward();optimizer.step()
    if ((epoch+1)%2000)==0:
        train_loss,valid_loss=estimate()
        print(f"Epoch:{epoch+1} Train_Loss:{train_loss:.2f} Valid_Loss:{valid_loss:.2f}")
        w.write(f"Epoch:{epoch+1} Train_Loss:{train_loss} Valid_Loss:{valid_loss}\n")
#存储模型
torch.save(model.state_dict(),"model/ssl1-0.pth")