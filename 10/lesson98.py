"""
問題96のプロンプトに対して、正解の感情ラベルを含むテキストを望ましい応答、
間違った感情ラベルを含むテキストを望ましくない応答として、
事前学習済み言語モデルを選好チューニング (preference tuning) を実施せよ。
選好チューニングのアルゴリズムとしては、
近傍方策最適化 (PPO: Proximal Policy Optimization) や
直接選好最適化 (DPO: Direct Preference Optimization) などが考えられる。
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,GPT2Model
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from trl import DPOTrainer,DPOConfig

# GPUに移動
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ読み込み
train_df=pd.read_csv("../SST-2/train.tsv",sep="\t")
dev_df=pd.read_csv("../SST-2/dev.tsv",sep="\t")

# モデルとトークナイザー
model=GPT2Model("openai-community/gpt2-medium")
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token # padトークンの定義

# データセット
class CustomDataset(Dataset):
    def __init__(self,df,tokenizer,max_len=128):
        self.texts=df["sentence"].tolist()
        self.labels=df["label"].tolist()
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        encoding=self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "texts":self.texts[idx],
            "labels":torch.tensor(self.labels[idx],dtype=torch.long)
        }
train_dataset=CustomDataset(train_df,tokenizer)
dev_dataset=CustomDataset(dev_df,tokenizer)

# Collate関数
def CustomCollate(batch):
    prompts=[]
    chosen=[]
    rejected=[]
    for item in batch:
        text=item["texts"]
        label=item["labels"].item()
        prompt = f"Review: {text}\nSentiment:"
        prompts.append(prompt)
        if label==1:
            chosen.append(" positive")
            rejected.append(" negative")
        else:
            chosen.append(" negative")
            rejected.append(" positive")
        return{
            "prompts":prompts,
            "chosen":chosen,
            "rejected":rejected
        }

# DPO
config=DPOConfig(
    beta=0.1,
    learning_rate=1e-5,
    batch_size=2
)

trainer=DPOTrainer(
    model,
    ref_model=None,
    args=config,
    beta=config.beta,
    train_dataset=train_dataset,
    data_collator=CustomCollate,
    tokenizer=tokenizer,
)

trainer.train()