import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, qkv_dim,embed_dim):
        super(SingleHeadAttention, self).__init__()
        self.qkv_dim=qkv_dim
        self.embed_dim = embed_dim
        self.W_q = torch.nn.Linear(qkv_dim, embed_dim, bias=False)
        self.W_k = torch.nn.Linear(qkv_dim, embed_dim, bias=False)
        self.W_v = torch.nn.Linear(qkv_dim, embed_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query) #Q,KVe
        K = self.W_k(key)   #KV,KVe
        V = self.W_v(value) #KV,Ve
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.qkv_dim, dtype=torch.float32)) #Q,KV
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf')) #Q,KV
        attention_weights = F.softmax(attention_scores, dim=-1) #Q,KV
        attended_values = torch.matmul(attention_weights, V) #Q,Ve
        return attended_values, attention_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, qkv_size,embed_size, num_heads,output_size):
        super(MultiHeadAttention, self).__init__()
        self.qkv_size=qkv_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention_heads = torch.nn.ModuleList([SingleHeadAttention(qkv_size,embed_size) for _ in range(num_heads)])
        self.fc_out = torch.nn.Linear(num_heads * embed_size, output_size)

    def forward(self, query, key, value, mask=None):
        head_outputs = [attention(query, key, value, mask)[0] for attention in self.attention_heads]
        concatenated_output = torch.cat(head_outputs, dim=-1)
        output = self.fc_out(concatenated_output)
        return output


class Encoder(nn.Module):
    def __init__(self,shape0, latent_dim=10,kl_coef=0.000001,BASENUM=256,embed_size=8,device='cpu'):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.device=device
        self.kl_coef=kl_coef
        self.trans_enc=MultiHeadAttention(1,embed_size, 1, embed_size)
        self.dense1=nn.Linear(shape0*embed_size,BASENUM*embed_size)
        self.bn1=nn.BatchNorm1d(BASENUM*embed_size)

        self.dense2=nn.Linear(BASENUM*embed_size,BASENUM//2*embed_size)
        self.bn2=nn.BatchNorm1d(BASENUM//2*embed_size)

        self.dense3=nn.Linear(BASENUM//2*embed_size,BASENUM//4*embed_size)
        self.bn3=nn.BatchNorm1d(BASENUM//4*embed_size)

        self.dense4=nn.Linear(BASENUM//4*embed_size,BASENUM//8*embed_size)
        self.bn4=nn.BatchNorm1d(BASENUM//8*embed_size)

        self.mu=nn.Linear(BASENUM//8*embed_size, latent_dim)
        self.logvar=nn.Linear(BASENUM//8*embed_size, latent_dim)
        self.kl = 0
    def reparameterize(self, mu , logvar):
        std = torch.exp(logvar*0.5)
        eps = torch.randn_like(std).to(self.device)
        z = mu + eps * std
        return z
    def forward(self, x):
        bn=x.size(0)
        x=x.view(bn,-1,1)
        x=self.trans_enc(x,x,x)
        x=x.view(bn,-1)
        print(x.size())
        x=F.relu(self.bn1(self.dense1(x)))
        x=F.relu(self.bn2(self.dense2(x)))
        x=F.relu(self.bn3(self.dense3(x)))
        x=F.relu(self.bn4(self.dense4(x)))
        mu =  self.mu(x)
        logvar = self.logvar(x)
        z=self.reparameterize(mu , logvar)
        self.kl = 0.5*(logvar.exp() + mu**2 - logvar - 1).sum()*self.kl_coef
        return z


class Decoder(nn.Module):
    def __init__(self,shape0, latent_dim=8,BASENUM=512,device='cpu'):
        super(Decoder, self).__init__()
        self.dense1=nn.Linear(latent_dim,BASENUM//8)
        self.bn1=nn.BatchNorm1d(BASENUM//8)

        self.dense2=nn.Linear(BASENUM//8,BASENUM//4)
        self.bn2=nn.BatchNorm1d(BASENUM//4)

        self.dense3=nn.Linear(BASENUM//4,BASENUM//2)
        self.bn3=nn.BatchNorm1d(BASENUM//2)

        self.dense4=nn.Linear(BASENUM//2,BASENUM)
        self.bn4=nn.BatchNorm1d(BASENUM)
        
        self.out=nn.Linear(BASENUM,shape0)

    def forward(self, z):
        z = F.relu(self.bn1(self.dense1(z)))
        z = F.relu(self.bn2(self.dense2(z)))
        z = F.relu(self.bn3(self.dense3(z)))
        z = F.relu(self.bn4(self.dense4(z)))
        z = self.out(z)
        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self,shape0, latent_dims=10,kl_coef=0.000001,BASENUM=256,embed_size=4,device='cpu'):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(shape0,latent_dims,kl_coef,BASENUM,embed_size,device).to(device)
        self.decoder = Decoder(shape0,latent_dims,BASENUM*embed_size,device).to(device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



def train(model,opt,loss_fn,train_loader,test_loader=None,device='cpu',n_epochs=100,save=False):
    model.train()
    for epoch in range(n_epochs):
        train_loss1=0
        train_loss2=0
        model.train()
        max_loss=1e9
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
            if save:
                if train_loss1/len(train_loader)+train_loss2/len(train_loader)<max_loss:
                    max_loss=train_loss1/len(train_loader)+train_loss2/len(train_loader)
                    torch.save(model.state_dict(), "best_model.pt")
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