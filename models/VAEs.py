import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Encoder(nn.Module):
    def __init__(self,shape0, latent_dim=10,kl_coef=0.000001,BASENUM=512,device='cpu'):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.device=device
        self.kl_coef=kl_coef
        self.dense1=nn.Linear(shape0,BASENUM)
        self.bn1=nn.BatchNorm1d(BASENUM)

        self.dense2=nn.Linear(BASENUM,BASENUM//2)
        self.bn2=nn.BatchNorm1d(BASENUM//2)

        self.dense3=nn.Linear(BASENUM//2,BASENUM//4)
        self.bn3=nn.BatchNorm1d(BASENUM//4)

        self.mu=nn.Linear(BASENUM//4, latent_dim)
        self.logvar=nn.Linear(BASENUM//4, latent_dim)
        self.kl = 0
    def reparameterize(self, mu , logvar):
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z
    def forward(self, x):
        bn=x.size(0)
        x=F.relu(self.bn1(self.dense1(x)))
        x=F.relu(self.bn2(self.dense2(x)))
        x=F.relu(self.bn3(self.dense3(x)))
        mu =  self.mu(x)
        logvar = self.logvar(x)
        z=self.reparameterize(mu , logvar)
        self.kl = 0.5*(logvar.exp() + mu**2 - logvar - 1).sum()*self.kl_coef
        return z


class Decoder(nn.Module):
    def __init__(self,shape0, latent_dim=8,BASENUM=512,device='cpu'):
        super(Decoder, self).__init__()
        self.dense1=nn.Linear(latent_dim,BASENUM//4)
        self.bn1=nn.BatchNorm1d(BASENUM//4)

        self.dense2=nn.Linear(BASENUM//4,BASENUM//2)
        self.bn2=nn.BatchNorm1d(BASENUM//2)

        self.dense3=nn.Linear(BASENUM//2,BASENUM)
        self.bn3=nn.BatchNorm1d(BASENUM)
        
        self.out=nn.Linear(BASENUM,shape0)

    def forward(self, z):
        z = F.relu(self.bn1(self.dense1(z)))
        z = F.relu(self.bn2(self.dense2(z)))
        z = F.relu(self.bn3(self.dense3(z)))
        z = self.out(z)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self,shape0, latent_dims=10,kl_coef=0.000001,BASENUM=512,device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(shape0,latent_dims,kl_coef,BASENUM,device).to(device)
        self.decoder = Decoder(shape0,latent_dims,BASENUM,device).to(device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def train(model,opt,loss_fn,train_loader,test_loader=None,device='cpu',n_epochs=100):
    model.train()
    for epoch in range(n_epochs):
        train_loss1=0
        train_loss2=0
        model.train()
        for batch in tqdm.tqdm(train_loader):
            x = batch['x'].to(device) # GPU
            c = batch['c'].to(device) # GPU
            opt.zero_grad()
            x_hat = model(x)
            loss1=loss_fn(x_hat,x)
            loss2=model.encoder.kl
            loss = loss1 + loss2
            train_loss1+=loss1.detach().cpu().numpy()
            train_loss2+=loss2.detach().cpu().numpy()
            loss.backward()
            opt.step()
        print(f"TRAIN: EPOCH {epoch}: MSE: {train_loss1/len(train_loader)}, KL_LOSS: {train_loss2/len(train_loader)}")


def encode(model,dataset,device='cpu'):
    model.eval()
    encoded_x=[]
    cs=[]
    for rec in tqdm.tqdm(dataset):
        x = rec['x'].reshape(1,-1).to(device) # GPU
        c = rec['c'].reshape(1,).to(device) # GPU
        encoded_x.append(model.encoder(x).cpu().detach().numpy())
        cs.append(c.cpu().detach().numpy())
    encoded_x=np.concatenate(encoded_x,axis=0)
    encoded_x=(encoded_x-encoded_x.mean(axis=0,keepdims=True))/encoded_x.std(axis=0,keepdims=True)
    cs=np.concatenate(cs,axis=0)
    df_to_be_shown=pd.DataFrame(encoded_x,columns=[f'f{i}' for i in range(encoded_x.shape[1])])
    df_to_be_shown['control']=cs
    return df_to_be_shown