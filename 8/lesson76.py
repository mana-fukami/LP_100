"""
問題75のパディングの処理を活用して、ミニバッチでモデルを学習せよ。
また、学習したモデルの開発セットにおける正解率を求めよ。
"""
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lesson70 import embedding_matrix,token_to_id,id_to_token

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

# カスタムコラート関数
def collate(batch):
    # トークン列の長さでソートする
    batch.sort(key=lambda x:len(x["input_ids"]),reverse=True)
    labels = torch.stack([item["label"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]
    features=torch.stack([item["feature_vec"] for item in batch])

    # input_idsをパディングして同じ長さに揃える
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {"input_ids": padded_input_ids,"label": labels, "feature_vec": features}

train_dataset=CustomDataset(train_df)
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True, collate_fn=collate)
dev_dataset=CustomDataset(dev_df)
dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True, collate_fn=collate)

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

#損失関数と最適化手法
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        features=batch["feature_vec"].to(device)
        labels=batch["label"].to(device).view(-1,1) # ラベルの形状を[64,1]に変換。デフォルトが[64]になっている

        outputs=model(features)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# モデルの評価
model.eval()
correct=0
total=0

with torch.no_grad(): #計算の高速化のために勾配計算を無効にする
    for batch in dev_dataloader:
        features=batch["feature_vec"]
        labels=batch["label"].to(device).view(-1,1)

        # モデルの出力を取得
        outputs=model(features)

        # 0.5を閾値として二値分類
        predicted=(outputs>0.5).float()

        #正解数のカウント
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

# 正解率を計算
accuracy=100*correct/total
print(f"正解率: {accuracy:.2f}%")
# 実行結果
"""
Using cpu device
Epoch 1/10, Loss: 0.5245800018310547
Epoch 2/10, Loss: 0.35233908891677856
Epoch 3/10, Loss: 0.2389199435710907
Epoch 4/10, Loss: 0.3887180984020233
Epoch 5/10, Loss: 0.3741491436958313
Epoch 6/10, Loss: 0.5937873721122742
Epoch 7/10, Loss: 0.4696057140827179
Epoch 8/10, Loss: 0.3883538842201233
Epoch 9/10, Loss: 0.6466307640075684
Epoch 10/10, Loss: 0.5669580101966858
正解率: 79.59%
"""