"""
General Language Understanding Evaluation (GLUE) ベンチマークで配布されている
Stanford Sentiment Treebank (SST) から訓練セット(train.tsv)と開発セット(dev.tsv)の
テキストと極性ラベルと読み込み、さらに全てのテキストはトークン列に変換せよ。
"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

train_tsv=open("SST-2/train.tsv","r")
train_df=pd.read_csv(train_tsv,sep="\t")

dev_tsv=open("SST-2/dev.tsv","r")
dev_df=pd.read_csv(dev_tsv,sep="\t")

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

class CustomDataset(Dataset):
    def __init__(self,data_df):
        self.data=[]
        for i in range(data_df.shape[0]):
            d={}
            # テキストの読み込み
            text=data_df.loc[i,"sentence"]
            # 極性ラベルの読み込み
            label=data_df.loc[i,"label"]
            d["text"]=text
            d["label"]=torch.tensor(label,dtype=torch.float32)
            # テキストをテンソル形式のトークン列に変換
            d["tokens"]=tokenizer(text,return_tensors="pt")
            self.data.append(d)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

train_dataset=CustomDataset(train_df)
dev_dataset=CustomDataset(dev_df)

def show_result():
    print(train_dataset.__getitem__(10))
    print("--------------------")
    print(dev_dataset.__getitem__(10))

#show_result()
"""
{'text': 'goes to absurd lengths ', 'label': tensor(0.), 'tokens': {'input_ids': tensor([[  101,  3632,  2000, 18691, 10742,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}}
--------------------
{'text': 'the mesmerizing performances of the leads keep the film groundeilm grounded and keep the audience riveted . ', 'label': tensor(1.), 'tokens': {'input_ids': tensor([[  101,  1996,  2033,  6491, 11124,  6774,  4616,  1997,  1996,  5260,
          2562,  1996,  2143, 16764,  1998,  2562,  1996,  4378, 15544, 19510,
          2098,  1012,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}}
"""
# 見やすく整理
"""
{
    'text': 'goes to absurd lengths ',
    'label': tensor(0.),
    'tokens': {
        'input_ids': tensor([[  101,  3632,  2000, 18691, 10742,   102]]),
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
    }
}
--------------------
{
    'text': 'the mesmerizing performances of the leads keep the film groundeilm grounded and keep the audience riveted . ',
    'label': tensor(1.),
    'tokens': {
        'input_ids': tensor([[  101,  1996,  2033,  6491, 11124,  6774,  4616,  1997,  1996,  5260,  2562,  1996,  2143, 16764,  1998,  2562,  1996,  4378, 15544, 19510,  2098,  1012,   102]]),
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    }
}
"""