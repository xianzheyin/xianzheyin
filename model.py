import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import random
#多层前馈神经网络
class FeedForwardNetwork(nn.Module):
    def __init__(self,paras):
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(paras["d_model"],4*paras["d_model"]),
            nn.ReLU(),
            nn.Linear(4*paras["d_model"],paras["d_model"]),
            nn.Dropout(paras["dropout"])
        )
    def forward(self,x):
        return self.ffn(x)
#单头注意力机制
class Attention(nn.Module):
    def __init__(self,paras):
        super().__init__()
        self.Wq=nn.Linear(paras["d_model"],paras["head_size"],bias=False)
        self.Wk=nn.Linear(paras["d_model"],paras["head_size"],bias=False)
        self.Wv=nn.Linear(paras["d_model"],paras["head_size"],bias=False)
        self.Dropout=nn.Dropout(paras["dropout"])
        self.head_size=paras["head_size"]
    def forward(self,x):
        q=self.Wq(x);k=self.Wk(x);v=self.Wv(x)
        output=(q@k.transpose(-2,-1))/math.sqrt(self.head_size)
        output=F.softmax(output,dim=-1)
        output=self.Dropout(output)
        output=output@v
        return output
#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,paras):
        super().__init__()
        self.heads=nn.ModuleList([Attention(paras) for _ in range(paras["head_num"])])
        self.Wo=nn.Linear(paras["d_model"],paras["d_model"])
        self.Dropout=nn.Dropout(paras["dropout"])
    def forward(self,x):
        output=torch.cat([head(x) for head in self.heads],dim=-1)
        output=self.Dropout(self.Wo(output))
        return output
#Transformer单块
class TransformerBlock(nn.Module):
    def __init__(self,paras):
        super().__init__()
        self.ln1=nn.LayerNorm(paras["d_model"])
        self.ln2=nn.LayerNorm(paras["d_model"])
        self.mha=MultiHeadAttention(paras)
        self.ffn=FeedForwardNetwork(paras)
    def forward(self,x):
        x=x+self.mha(self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x
#模型
class Model(nn.Module):
    def __init__(self,d_model,head_num,block_num,dropout,head_size,embedding_len,mask_percent):
        super().__init__()
        self.paras={"d_model":d_model,"head_num":head_num,"block_num":block_num,
                    "dropout":dropout,"head_size":head_size,"embedding_len":embedding_len,
                    "mask_percent":mask_percent}
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.TransformerBlock=nn.Sequential(
            *([TransformerBlock(self.paras) for _ in range(block_num)]+[nn.LayerNorm(d_model)])).to(self.device)
        self.embedding_ele=nn.Linear(embedding_len,d_model,device=self.device)
        self.embedding_snr=nn.Linear(embedding_len,d_model,device=self.device)
        self.embedding_fre=nn.Linear(1,embedding_len,device=self.device)
        self.unbedding=nn.Linear(d_model,embedding_len,device=self.device)
        self.decoder1=nn.Linear(d_model,1,device=self.device)
        self.decoder2=nn.Linear(d_model,1,device=self.device)
    def forward(self,snr,ele,fre):
        snr=torch.tensor(snr,dtype=torch.float32,device=self.device)
        ele=torch.tensor(ele,dtype=torch.float32,device=self.device)
        fre=torch.tensor(fre,dtype=torch.float32,device=self.device)
        B,L=ele.shape
        #插入分类头
        ele=torch.cat([ele,torch.tensor(torch.zeros((B,10)),dtype=torch.float32,device=self.device)],dim=1)
        snr=torch.cat([snr,self.embedding_fre(fre)],dim=1)
        #编码
        embed_ele=ele.reshape((B,-1,self.paras["embedding_len"]))
        embed_snr=snr.reshape((B,-1,self.paras["embedding_len"]))
        input_snr=self.embedding_ele(embed_ele)+self.embedding_snr(embed_snr)
        outputs=self.TransformerBlock(input_snr)
        outputs=self.unbedding(outputs).to("cpu")
        outputs=outputs.reshape((B,L+10))
        return outputs[:,:L]
    def forward1(self,snr,ele,fre):
        snr=torch.tensor(snr,dtype=torch.float32,device=self.device)
        ele=torch.tensor(ele,dtype=torch.float32,device=self.device)
        fre=torch.tensor(fre,dtype=torch.float32,device=self.device)
        B,L=ele.shape
        #插入分类头
        ele=torch.cat([ele,torch.tensor(torch.zeros((B,10)),dtype=torch.float32,device=self.device)],dim=1)
        snr=torch.cat([snr,self.embedding_fre(fre)],dim=1)
        #编码
        embed_ele=ele.reshape((B,-1,self.paras["embedding_len"]))
        embed_snr=snr.reshape((B,-1,self.paras["embedding_len"]))
        input_snr=self.embedding_ele(embed_ele)+self.embedding_snr(embed_snr)
        outputs=self.TransformerBlock(input_snr)
        return self.decoder1(outputs[:,-1,:]).reshape(-1).to('cpu')
    def forward2(self,snr,ele,fre):
        snr=torch.tensor(snr,dtype=torch.float32,device=self.device)
        ele=torch.tensor(ele,dtype=torch.float32,device=self.device)
        fre=torch.tensor(fre,dtype=torch.float32,device=self.device)
        B,L=ele.shape
        #插入分类头
        ele=torch.cat([ele,torch.tensor(torch.zeros((B,10)),dtype=torch.float32,device=self.device)],dim=1)
        snr=torch.cat([snr,self.embedding_fre(fre)],dim=1)
        #编码
        embed_ele=ele.reshape((B,-1,self.paras["embedding_len"]))
        embed_snr=snr.reshape((B,-1,self.paras["embedding_len"]))
        input_snr=self.embedding_ele(embed_ele)+self.embedding_snr(embed_snr)
        outputs=self.TransformerBlock(input_snr)
        score=self.decoder2(outputs[:,-1,:])
        score=score.reshape(-1).to('cpu')
        return self.decoder1(outputs[:,-1,:]).reshape(-1).to('cpu'),score