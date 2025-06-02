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

input_dim=embedding_matrix.shape[1]
model = NeuralNetwork(input_dim).to(device)

def show_result():
    print(model)

#show_result()
# 実行結果
"""
Using cpu device
NeuralNetwork(
  (linear): Linear(in_features=300, out_features=1, bias=True)      
)
"""