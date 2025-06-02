"""
単語埋め込みの平均ベクトルでテキストの特徴ベクトルを表現し、
重みベクトルとの内積でポジティブ及びネガティブを分類する
ニューラルネットワーク（ロジスティック回帰モデル）を設計せよ。
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lesson70 import embedding_matrix,token_to_id,id_to_token
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

train_tsv=open("SST-2/train.tsv","r")
train_df=pd.read_csv(train_tsv,sep="\t")

dev_tsv=open("SST-2/dev.tsv","r")
dev_df=pd.read_csv(dev_tsv,sep="\t")

class CustomDataset(Dataset):
    def __init__(self,data_df):
        token_id=self.text_to_token_id(data_df)
        self.token_id_feature=self.add_average_vec(token_id)

    def __len__(self):
        return len(self.token_id_feature)

    def __getitem__(self,idx):
        return self.token_id_feature[idx]

    def text_to_token_id(self,df):
        token_id=[]
        for i in range(df.shape[0]):
            data={}
            label=df.loc[i,"label"]
            text=df.loc[i,"sentence"]
            data["text"]=text
            data["label"]=torch.tensor(label,dtype=torch.float32)
            ids=self.get_token_id(text)
            if ids!=[]:
                data["input_ids"]=torch.tensor(ids)
                token_id.append(data)
        return token_id

    def get_token_id(self,text):
        words=text.split(" ")
        ids=[]
        for word in words:
            if word in token_to_id.keys():
                id=token_to_id[word]
                ids.append(id)
        return ids

    def add_average_vec(self,data_list):
        for data in data_list:
            ids=data["input_ids"]
            vectors=[embedding_matrix[id] for id in ids]
            average_vec=np.mean(vectors,axis=0)
            data["feature_vec"]=torch.tensor(average_vec,dtype=torch.float32)
        return data_list

train_dataset=Dataset(train_df)
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
dev_dataset=Dataset(dev_df)
dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self,input_dim):
        super(NeuralNetwork,self).__init__()
        self.linear=nn.Linear(input_dim,1)

    def forward(self,x):
        logits=self.linear(x)
        probs=torch.sigmoid(logits)
        return probs

def show_result():
    model = NeuralNetwork().to(device)
    print(model)

show_result()
# 実行結果
"""

"""