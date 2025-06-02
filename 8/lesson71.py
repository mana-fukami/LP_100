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
            data["label"]=torch.tensor([label],dtype=torch.float32)
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
def custom_collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]

    # input_idsをパディングして同じ長さに揃える
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {"text": texts, "label": labels, "input_ids": padded_input_ids}

train_dataset=CustomDataset(train_df)
train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True,collate_fn=custom_collate_fn)
dev_dataset=CustomDataset(dev_df)
dev_dataloader=DataLoader(dev_dataset,batch_size=2,shuffle=True,collate_fn=custom_collate_fn)

def show_result():
    train_feature=next(iter(train_dataloader))
    print(train_feature)
    print("--------------------")
    dev_feature=next(iter(dev_dataloader))
    print(dev_feature)

show_result()
# 実行結果
"""
{'text': ['very much worth ', 'outdated '], 'label': tensor([[1.],
        [0.]]), 'input_ids': tensor([[  139,   151,  1070],
        [13000,     0,     0]])}
--------------------
{'text': ['complete lack of originality , cleverness or even visible effort ', 'one of those energetic surprises , an original that pleases almost everyone who sees it . '], 'label': tensor([[0.],        
        [1.]]), 'input_ids': tensor([[  924,  1384, 27549, 81316,    30,   156,  5358,   798,     0,     0,
             0,     0,     0],
        [   46,   134, 12156,  9638,    28,  1413,     4, 40862,   608,   891,
            32,  2660,    16]])}
"""