"""
事前学習済み言語モデルでテキストをベクトルで表現（エンコード）し、そのベクトルにフィードフォワード層を通すことで極性ラベルを予測するモデルを学習せよ。
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,GPT2Model
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# GPUに移動
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# データ読み込み
train_df=pd.read_csv("../SST-2/train.tsv",sep="\t")
dev_df=pd.read_csv("../SST-2/dev.tsv",sep="\t")

# トークナイザー
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token # padトークンの定義

# データセットを定義
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
            "input_ids":encoding["input_ids"].squeeze(0),
            "attention_mask":encoding["attention_mask"].squeeze(0),
            "label":torch.tensor(self.labels[idx],dtype=torch.long)
        }
train_dataset=CustomDataset(train_df,tokenizer)
dev_dataset=CustomDataset(dev_df,tokenizer)

# データローダー
train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
dev_loader=DataLoader(dev_dataset,batch_size=16,shuffle=True)

# モデル定義
class SentimentClassifier(nn.Module):
    def __init__(self,encoder_name="openai-community/gpt2-medium",hidden_dim=256):
        super().__init__()
        self.model=GPT2Model.from_pretrained(encoder_name)
        self.model.config.pad_token_id = tokenizer.pad_token_id
        self.classifier=nn.Sequential(
            nn.Linear(self.model.config.hidden_size,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),# 過学習の防止
            nn.Linear(hidden_dim,2)
        )

    def forward(self,input_ids,attention_mask):
        outputs=self.model(input_ids=input_ids,attention_mask=attention_mask)
        cls_vec=outputs.last_hidden_state.mean(dim=1)
        return self.classifier(cls_vec)

# 学習準備
model=SentimentClassifier().to(device)
model = nn.DataParallel(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
loss_fn=nn.CrossEntropyLoss()

# 学習ループ
for epoch in range(3):
    model.train()
    total_loss=0
    for batch in tqdm(train_loader):
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)
        outputs=model(input_ids,attention_mask)
        loss=loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 評価
model.eval()
preds=[]
labels_all=[]

with torch.no_grad():
    for batch in dev_loader:
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)
        outputs=model(input_ids,attention_mask)
        pred=torch.argmax(outputs,dim=1)
        preds.extend(pred.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

acc=accuracy_score(labels_all,preds)
print(f"正解率: {acc:.4f}")
"""
Using cuda
/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/huggingface_hub/file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
100%|███████████████████████████████████████████████| 4210/4210 [46:55<00:00,  1.50it/s]
Epoch 1, Loss: 947.0114
100%|███████████████████████████████████████████████| 4210/4210 [46:40<00:00,  1.50it/s]
Epoch 2, Loss: 508.2394
100%|███████████████████████████████████████████████| 4210/4210 [46:40<00:00,  1.50it/s]
Epoch 3, Loss: 349.6616
正解率: 0.9220
"""