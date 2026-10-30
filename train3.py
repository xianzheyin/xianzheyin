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
dropout=0.1;learning_rate=1e-4*0.5;batch_size=144;embedding_len=10
num_epochs=400000;mask_percent=0.4;valid_epoch=100
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
    train_dataset[i]=cls.trackset1(train_data)
    train_loader[i]=DataLoader(train_dataset[i],batch_size=batch_size,shuffle=True)
    valid_dataset[i]=cls.trackset1(valid_data)
    valid_loader[i]=DataLoader(valid_dataset[i],batch_size=batch_size,shuffle=True)
#加载模型
model=model.Model(d_model,head_num,block_num,dropout,d_model//head_num,embedding_len,mask_percent)
model.load_state_dict(torch.load("model/confidence1-0.pth"));model.to(device);model.train()
criterion=nn.MSELoss();optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
train_size_max=float(max(train_size))
for i in range(34):
    train_size0[i]=float(train_size[i]);train_size0[i]=train_size0[i]/train_size_max
#精度评估
def estimate():
    with torch.no_grad():
        model.eval()
        #训练集
        train_loss0=[]
        for _ in range(valid_epoch):
            p=random.randint(0,33)
            while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
            inputs_snr,inputs_ele,ref_e,inputs_fre=next(iter(train_loader[p]))
            ref_e=torch.tensor(ref_e,dtype=torch.float32)
            outputs,score=model.forward2(inputs_snr,inputs_ele,inputs_fre)
            lo=criterion(ref_e,outputs).clone().detach().numpy()
            if abs(lo)<=0.4:train_loss0.append(lo)
        train_loss0=np.array(train_loss0)
        train_loss=np.sqrt(np.mean(train_loss0))
        #验证集
        valid_loss0=[]
        for _ in range(valid_epoch):
            p=random.randint(0,33)
            while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
            inputs_snr,inputs_ele,ref_e,inputs_fre=next(iter(valid_loader[p]))
            ref_e=torch.tensor(ref_e,dtype=torch.float32)
            outputs,score=model.forward2(inputs_snr,inputs_ele,inputs_fre)
            lo=criterion(ref_e,outputs).clone().detach().numpy()
            if abs(lo)<=0.4:valid_loss0.append(lo)
        valid_loss0=np.array(valid_loss0)
        valid_loss=np.sqrt(np.mean(valid_loss0))
        model.train()
        return train_loss,valid_loss
#训练
list_train_loss=[];list_valid_loss=[]
w=open("output/Loss1-03.txt","w")
for epoch in range(num_epochs):
    if epoch<=num_epochs*0.1 and epoch%100==0:optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate*((epoch+1)*1.0/(num_epochs*0.1)))
    if epoch==num_epochs*0.4:optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate*0.5)
    if epoch==num_epochs*0.6:optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate*0.12)
    if epoch==num_epochs*0.8:optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate*0.03)
    p=random.randint(0,33)
    while train_size0[p]<random.uniform(0,1) or train_size[p]==0 or valid_size[p]==0:p=random.randint(0,33)
    inputs_snr,inputs_ele,ref_e,inputs_fre=next(iter(train_loader[p]))
    ref_e=torch.tensor(ref_e,dtype=torch.float32)
    outputs,log_var=model.forward2(inputs_snr,inputs_ele,inputs_fre)
    variance=torch.exp(log_var)
    error=ref_e-outputs
    abs_error=torch.abs(error)
    weights=torch.ones_like(abs_error)
    weights=torch.where(abs_error>0.5,torch.zeros_like(weights),weights)
    mid_mask=(abs_error>0.3)&(abs_error<=0.5)
    weights=torch.where(mid_mask,(0.5-abs_error.clone().detach())/0.2,weights)
    sample_loss=(error**2)/variance+log_var
    if weights.sum()==0:continue
    loss=(weights*sample_loss).sum()/weights.sum()
    optimizer.zero_grad();loss.backward();optimizer.step()
    if ((epoch+1)%4000)==0:
        train_loss,valid_loss=estimate()
        print(f"Epoch:{epoch+1} Train_Loss:{train_loss:.4f} Valid_Loss:{valid_loss:.4f}")
        w.write(f"Epoch:{epoch+1} Train_Loss:{train_loss} Valid_Loss:{valid_loss}\n")
#存储模型
torch.save(model.state_dict(),"model/ft1-0.pth")