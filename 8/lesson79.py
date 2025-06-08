"""
ニューラルネットワークのアーキテクチャを自由に変更し、モデルを学習せよ。
また、学習したモデルの開発セットにおける正解率を求めよ。
例えば、テキストの特徴ベクトル(単語埋め込みの平均ベクトル)
に対して多層のニューラルネットワークを通したり、
畳み込みニューラルネットワーク(CNN; Convolutional Neural Network)
や再帰型ニューラルネットワーク(RNN; Recurrent Neural Network)
などのモデルの学習に挑戦するとよい。
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

class CNN(nn.Module):
    def __init__(self,input_dim,num_filters=100,kernel_size=1):
        # 今回は1次元の平均ベクトルを入力するので、カーネルサイズは1
        super(CNN,self).__init__()
        # 1次元の畳み込みを定義
        self.conv=nn.Conv1d(in_channels=input_dim,out_channels=num_filters,kernel_size=kernel_size)
        # 活性化関数を定義
        self.relu=nn.ReLU()
        # 可変長の入力に対応
        self.pool=nn.AdaptiveMaxPool1d(1)
        # 全結合して出力を1次元に
        self.fc=nn.Linear(num_filters,1)

    def forward(self,x):
        # 入力をCNNに適する形に
        x=x.unsqueeze(1).permute(0,2,1)  # [batch_size, input_dim, seq_len]に変換
        # モデルを通す
        x=self.conv(x)
        x=self.relu(x)
        x=self.pool(x).squeeze(-1)
        logits=self.fc(x)
        probs=torch.sigmoid(logits)
        return probs


input_dim=embedding_matrix.shape[1]
model = CNN(input_dim).to(device)
model=nn.DataParallel(model) # 並列処理ができるようにラップする
model=model.to(device)

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
# CNN
"""
Using cpu device
Epoch 1/10, Loss: 0.6377128958702087
Epoch 2/10, Loss: 0.37279942631721497
Epoch 3/10, Loss: 0.26339197158813477
Epoch 4/10, Loss: 0.21608556807041168
Epoch 5/10, Loss: 0.19463324546813965
Epoch 6/10, Loss: 0.1172608733177185
Epoch 7/10, Loss: 0.11716315895318985
Epoch 8/10, Loss: 0.13027139008045197
Epoch 9/10, Loss: 0.12842309474945068
Epoch 10/10, Loss: 0.1443399339914322
正解率: 80.62%
"""