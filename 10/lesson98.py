"""
問題96のプロンプトに対して、正解の感情ラベルをテキストの応答
として返すように事前学習済みモデルをファインチューニングせよ。
"""
import torch
from transformers import AutoTokenizer,GPT2LMHeadModel,Trainer,TrainingArguments
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# GPUに移動
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# データ読み込み
train_df=pd.read_csv("../SST-2/train.tsv",sep="\t")
dev_df=pd.read_csv("../SST-2/dev.tsv",sep="\t")

# モデルとトークナイザー
model=GPT2LMHeadModel.from_pretrained("openai-community/gpt2-medium")
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token # padトークンの定義

# 学習用テキスト作成
def format_sst2(example):
    text=example["sentence"]
    label="positive" if example["label"]==1 else "negative"
    return f"Review: {text}\nSentiment: {label}"

train_texts = [format_sst2(row) for _, row in train_df.iterrows()]
dev_texts   = [format_sst2(row) for _, row in dev_df.iterrows()]

# PyTorch Datasetクラスを定義
def shift_labels(input_ids):
    labels=input_ids.clone()
    labels[labels==tokenizer.pad_token_id]=-100
    return labels

class SST2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = shift_labels(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# データセット作成
train_dataset = SST2Dataset(train_texts, tokenizer)
dev_dataset = SST2Dataset(dev_texts, tokenizer)

# DataLoaderを使用
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=8)

# モデルの設定
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model=model.to(device)

# 学習準備
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 学習ループ
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 評価
model.eval()
preds = []
labels_all = []

with torch.no_grad():
    for batch in dev_loader:
        input_ids=batch["input_ids"].to(device)
        attention_mask=batch["attention_mask"].to(device)
        labels=batch["label"].to(device)
        outputs=model(input_ids,attention_mask)
        pred=torch.argmax(outputs,dim=1)
        preds.extend(pred.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

acc = accuracy_score(labels_all, preds)
print(f"Accuracy: {acc:.4f}")
# モデル保存
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")
"""
Using cuda
/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/huggingface_hub/file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
100%|█████████████████████████████████████████████| 8419/8419 [1:25:55<00:00,  1.63it/s]
Epoch 1, Loss: 20552.1245
100%|█████████████████████████████████████████████| 8419/8419 [1:00:45<00:00,  2.31it/s]
Epoch 2, Loss: 16437.4929
100%|█████████████████████████████████████████████| 8419/8419 [1:00:46<00:00,  2.31it/s]
Epoch 3, Loss: 13581.0065
/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/sklearn/metrics/_classification.py:98: UserWarning: The number of unique classes is greater than 50% of the number of samples.
  type_true = type_of_target(y_true, input_name="y_true")
Traceback (most recent call last):
  File "/data/student/f2210543/LP_100/10/lesson98.py", line 119, in <module>
    acc = accuracy_score(labels_all, preds)
  File "/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/sklearn/utils/_param_validation.py", line 218, in wrapper
    return func(*args, **kwargs)
  File "/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/sklearn/metrics/_classification.py", line 359, in accuracy_score
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
  File "/data/student/f2210543/LP_100/myenv2/lib/python3.10/site-packages/sklearn/metrics/_classification.py", line 117, in _check_targets
    raise ValueError("{0} is not supported".format(y_type))
ValueError: multiclass-multioutput is not supported
"""