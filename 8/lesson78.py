"""
問題77の学習において、単語埋め込みのパラメータも同時に更新する
ファインチューニングを導入せよ。
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
        self.token_id=self.text_to_token_id(data_df)

    def __len__(self):
        return len(self.token_id)

    def __getitem__(self,idx):
        return self.token_id[idx]

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

# カスタムコラート関数
def collate(batch):
    # トークン列の長さでソートする
    batch.sort(key=lambda x:len(x["input_ids"]),reverse=True)
    labels = torch.stack([item["label"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]

    # input_idsをパディングして同じ長さに揃える
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {"input_ids": padded_input_ids,"label": labels}

train_dataset=CustomDataset(train_df)
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True, collate_fn=collate)
dev_dataset=CustomDataset(dev_df)
dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True, collate_fn=collate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self,embedding_matrix):
        super(NeuralNetwork,self).__init__()
        # 単語埋め込みのパラメーターも学習する
        vocab_size,embedding_dim=embedding_matrix.shape
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight=nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.linear=nn.Linear(embedding_dim,1)

    def forward(self,input_ids):
        embedded=self.embedding(input_ids)
        avg_vec=embedded.mean(dim=1)
        logits=self.linear(avg_vec)
        probs=torch.sigmoid(logits)
        return probs

model = NeuralNetwork(embedding_matrix)
model=nn.DataParallel(model) # 並列処理ができるようにラップする
model=model.to(device)

#損失関数と最適化手法
criterion=nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

num_epochs=10
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        features=batch["input_ids"].to(device)
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
        features=batch["input_ids"].to(device)
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
Epoch 1/10, Loss: 0.15663787722587585
Epoch 2/10, Loss: 0.30829697847366333
Epoch 3/10, Loss: 0.2585207521915436
Epoch 4/10, Loss: 0.08939235657453537
Epoch 5/10, Loss: 0.04208642244338989
Epoch 6/10, Loss: 0.0693623498082161
Epoch 7/10, Loss: 0.06890696287155151
Epoch 8/10, Loss: 0.25348448753356934
Epoch 9/10, Loss: 0.046509385108947754
Epoch 10/10, Loss: 0.2628816068172455
正解率: 78.90%
"""