import sys
sys.path.append('..')
from data.data_reader import *
from models.VAEs import *


import os
import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if not os.path.exists('35775554.h5ad'):
    download_file('https://plus.figshare.com/ndownloader/files/35775554','35775554.h5ad')
adata_orig = sc.read_h5ad("35775554.h5ad")
adata_orig.X[adata_orig.X == float("inf")]=0
adata_orig.X[np.isnan(adata_orig.X)]=0

# adata_orig.obs['gene_name']=list(pd.Series(adata_orig.obs.index).apply(lambda x:x.split("_")[1]))
adata_orig.obs['id']=range(adata_orig.obs.shape[0])

def cosine_similarity(A):
  AAt=np.matmul(A,A.transpose())
  n_A=np.sqrt((A**2).sum(axis=1)).reshape(-1,1)
  n_A=np.matmul(n_A,n_A.transpose())
  return AAt/(n_A)

class X_dataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {'x':torch.tensor(self.data.X[idx]),'c':torch.tensor(self.data.obs.iloc[idx]['core_scale_factor'])}


dataset=X_dataset(adata_orig)
train_loader=DataLoader(dataset,batch_size=32,shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder=VariationalAutoencoder(dataset[0]['x'].shape[0],128,1e-11,4096,device)
opt = torch.optim.Adam(autoencoder.parameters(),lr=0.001)
loss_fn=torch.nn.MSELoss()


train(autoencoder,opt,loss_fn,train_loader,None,device,200,True)
