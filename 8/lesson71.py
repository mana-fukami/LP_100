"""
General Language Understanding Evaluation (GLUE) ベンチマークで配布されている
Stanford Sentiment Treebank (SST) をダウンロードし、
訓練セット(train.tsv)と開発セット(dev.tsv)のテキストと極性ラベルと読み込み、全てのテキストをトークンID列に変換せよ。
このとき、単語埋め込みの語彙でカバーされていない単語は無視し、トークン列に含めないことにせよ。
また、テキストの全トークンが単語埋め込みの語彙に含まれておらず、空のトークン列となってしまう事例は、
訓練セットおよび開発セットから削除せよ(このため、第7章の実験で得られた正解率と比較できなくなることに注意せよ)。
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lesson70 import embedding_matrix,token_to_id,id_to_token

train_tsv=open("SST-2/train.tsv","r")
train_df=pd.read_csv(train_tsv,sep="\t")

dev_tsv=open("SST-2/dev.tsv","r")
dev_df=pd.read_csv(dev_tsv,sep="\t")

class Dataset():
    def __init__(self,data_df):
        self.token_id=self.text_to_token_id(data_df)

    def __len__(self):
        return len(self.token_id)

    def __getitem__(self,idx):
        return self.token_id[idx]

    def text_to_token_id(self,df):
        token_id=[]
        dict={}
        for i in range(df.shape[0]):
            label=df.loc[i,"label"]
            text=df.loc[i,"sentence"]
            dict["text"]=text
            dict["label"]=torch.tensor([float(label)])
            ids=self.get_token_id(text)
            if ids!=[]:
                dict["input_ids"]=torch.tensor(ids)
                token_id.append(dict)
        return token_id

    def get_token_id(self,text):
        words=text.split(" ")
        ids=[]
        for word in words:
            if word in token_to_id.keys():
                id=token_to_id[word]
                ids.append(id)
        return ids

train_dataset=Dataset(train_df)
train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True)
dev_dataset=Dataset(dev_df)
dev_dataloader=DataLoader(dev_dataset,batch_size=2,shuffle=True)

def show_result():
    train_feature=next(iter(train_dataloader))
    print(train_feature)
    #print(train_labels)
    print("--------------------")
    dev_feature=next(iter(dev_dataloader))
    print(dev_feature)
    #print(dev_labels)

show_result()
# 実行結果
"""
{'text': ['this new jangle of noise , mayhem and stupidity must be a serious contender for the title . ', 'this new jangle of noise , mayhem and stupidity must be a serious contender for the title . '], 'label': tensor([[0.],
        [0.]]), 'input_ids': tensor([[    29,     66, 169108,   4702,  18028,  25799,    337,     17,    982,
           7607,      3,     12,    759],
        [    29,     66, 169108,   4702,  18028,  25799,    337,     17,    982,
           7607,      3,     12,    759]])}
--------------------
{'text': ["looking aristocratic , luminous yet careworn in jane hamilton 's exemplary costumes , rampling gives a performance that could not be improved upon . ' ", "looking aristocratic , luminous yet careworn in jane hamilton 's exemplary costumes , rampling gives a performance that could not be improved upon . ' "], 'label': tensor([[1.],
        [1.]]), 'input_ids': tensor([[   380,  54575,  44396,    507, 301929,      2, 269337, 442845,  17067,
          10358,   1337,    476,      4,     76,     14,     17,   1519,   1473],
        [   380,  54575,  44396,    507, 301929,      2, 269337, 442845,  17067,
          10358,   1337,    476,      4,     76,     14,     17,   1519,   1473]])}
"""