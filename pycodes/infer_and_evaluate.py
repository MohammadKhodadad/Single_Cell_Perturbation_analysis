import sys
sys.path.append('..')
from models.VAEs import *
from data.data_reader import *



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


adata_orig = sc.read_h5ad("35775554.h5ad")
adata_orig.X[adata_orig.X == float("inf")]=0
adata_orig.X[np.isnan(adata_orig.X)]=0


class X_dataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {'x':torch.tensor(self.data.X[idx]),'c':torch.tensor(self.data.obs.iloc[idx]['core_scale_factor'])}


dataset=X_dataset(adata_orig)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder=VariationalAutoencoder(dataset[0]['x'].shape[0],128,1e-11,4096,device)
autoencoder.load_state_dict(torch.load('best_model.pt'))



df_to_be_shown=encode(autoencoder,dataset,device)
df_to_be_shown['gene_name']=list(adata_orig.obs.gene)
df_to_be_shown=df_to_be_shown.groupby(['gene_name']).mean()
df_to_be_shown['gene_name']=df_to_be_shown.index
df_to_be_shown.reset_index(drop=True,inplace=True)
df_to_be_shown['id']=range(df_to_be_shown.shape[0])


def cosine_similarity(A):
  AAt=np.matmul(A,A.transpose())
  n_A=np.sqrt((A**2).sum(axis=1)).reshape(-1,1)
  n_A=np.matmul(n_A,n_A.transpose())
  output=AAt/(n_A)
  output[np.isnan(output)]=0
  return output

cos_sim_f=cosine_similarity(np.array(df_to_be_shown.drop(['control','gene_name','id'], axis=1,errors='ignore')))

similarity_matrix=np.zeros(cos_sim_f.shape)
similarity_db=hu_data_loader()

for gene_name in tqdm.tqdm(df_to_be_shown.gene_name.unique()):
    query=query_hu_data(similarity_db,gene_name)
    for q in query:
        if q in df_to_be_shown.gene_name.values:
            y_indices=df_to_be_shown[df_to_be_shown.gene_name==q].id
            x_indices=df_to_be_shown[df_to_be_shown.gene_name==gene_name].id
            for x_id in x_indices:
                for y_id in y_indices:
                    similarity_matrix[y_id,x_id]=1
                    similarity_matrix[x_id,y_id]=1

cos_sim_f_flatten=cos_sim_f.reshape(-1,)
similarity_matrix_flatten=similarity_matrix.reshape(-1,)
cos_sim_f_flatten1=cos_sim_f_flatten[similarity_matrix_flatten==1]
cos_sim_f_flatten0=cos_sim_f_flatten[similarity_matrix_flatten==0]


def get_recall(rate):
    qrate_down=np.quantile(cos_sim_f_flatten,rate)
    qrate_up=np.quantile(cos_sim_f_flatten,1-rate)
    pred_p=np.logical_or(cos_sim_f_flatten>qrate_up,cos_sim_f_flatten<qrate_down)
    pred_n=np.logical_and(cos_sim_f_flatten<qrate_up,cos_sim_f_flatten>qrate_down)
    tp=np.logical_and(pred_p,similarity_matrix_flatten==1).sum()
    fp=np.logical_and(pred_p,similarity_matrix_flatten==0).sum()
    fn=np.logical_and(pred_n,similarity_matrix_flatten==1).sum()
    return tp/(tp+fn)
def visualize_recal_vs_quantile(fig_name='output'):
    values=[]
    xs=[i*0.05 for i in range(10)]
    for i in xs:
        values.append(get_recall(i))
    temp_df=pd.DataFrame({'quantile':xs,'recall':values})
    fig=px.line(temp_df,x='quantile',y='recall',title='recall_vs_quantile',width=1000, height=400)
    fig.update_traces(mode='lines+text', text=list(map(lambda x:round(x,2),values)), textposition='top center')
    fig.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=10,  # Set the desired font size
        color="black"
    )
)
    # fig.show()
    fig.write_html(f"{fig_name}.html")
    


visualize_recal_vs_quantile()


choice=np.random.choice(cos_sim_f_flatten1.shape[0], 2048)
cos_sim_f_flatten1=cos_sim_f_flatten1[choice]
cos_sim_f_flatten1=pd.DataFrame(cos_sim_f_flatten1,columns=['correlations'])
fig=px.violin(cos_sim_f_flatten1, y='correlations',width=500, height=400,title="SIMILARS")
# fig.show()


choice=np.random.choice(cos_sim_f_flatten0.shape[0], 2048)
cos_sim_f_flatten0=cos_sim_f_flatten0[choice]
cos_sim_f_flatten0=pd.DataFrame(cos_sim_f_flatten0,columns=['correlations'])
fig=px.violin(cos_sim_f_flatten0, y='correlations',width=500, height=400,title="Not SIMILARS")
# fig.show()



print("Not SIMILARS MEAN:",cos_sim_f_flatten0.mean())
print("SIMILARS MEAN:",cos_sim_f_flatten1.mean())

