"""
問題87とは異なるアーキテクチャ（例えば[CLS]トークンを用いるか、各トークンの最大値プーリングを用いるなど）の分類モデルを設計し、
事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。検証セット上でファインチューニングされたモデルの正解率を計測せよ。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import AutoTokenizer,AutoModel,Trainer,TrainingArguments
from sklearn.metrics import accuracy_score

# GPUに移動
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# データセットの作成
class CustomDataset(Dataset):
    def __init__(self,data_df,tokenizer,max_length=128):
        self.texts=data_df["sentence"].tolist()
        self.labels=data_df["label"].tolist()
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text=self.texts[idx]
        label=self.labels[idx]
        encoding=self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":encoding["input_ids"].squeeze(),
            "attention_mask":encoding["attention_mask"].squeeze(),
            "labels":torch.tensor(label,dtype=torch.long)
        }

train_tsv=open("../SST-2/train.tsv","r")
train_df=pd.read_csv(train_tsv,sep="\t")

dev_tsv=open("../SST-2/dev.tsv","r")
dev_df=pd.read_csv(dev_tsv,sep="\t")

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset=CustomDataset(train_df,tokenizer)
dev_dataset=CustomDataset(dev_df,tokenizer)

# モデルの定義
class MaxPoolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 事前学習済みモデルの読み込み
        self.bert=AutoModel.from_pretrained("bert-base-uncased")
        self.dropout=nn.Dropout(0.1)
        self.classifier=nn.Linear(self.bert.config.hidden_size,2)
    
    def forward(self,input_ids,attention_mask,labels=None):
        # BERTの出力の取得
        outputs=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        # 各トークンの最大値をプーリング
        pooled_output=outputs.last_hidden_state.max(dim=1)[0]
        # ドロップアウトと分類
        pooled_output=self.dropout(pooled_output)
        logits=self.classifier(pooled_output)
        loss=None
        if labels is not None:
            loss_fn=nn.CrossEntropyLoss()
            loss=loss_fn(logits.view(-1,2),labels.view(-1))
        return {"loss": loss,"logits": logits}

model=MaxPoolClassifier()
model=model.to(device)

# Trainerの設定
training_args=TrainingArguments(
    output_dir="./results_maxpool",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    lr_scheduler_type="linear",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    remove_unused_columns=False,
    report_to="none"
)

# 評価指標
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy": accuracy_score(labels,preds)}

# Trainerで学習
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

trainer.train()

# 正解率を計算
print(f"正解率: {trainer.evaluate()}")
"""
正解率: {'eval_loss': 0.26920342445373535, 'eval_accuracy': 0.9174311926605505, 'eval_runtime': 2.9571, 'eval_samples_per_second': 294.886, 'eval_steps_per_second': 4.734, 'epoch': 3.0}
"""
"""
Using cuda device
/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/huggingface_hub/file_download.py:943: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  0%|                                                                                 | 0/3159 [00:00<?, ?it/s]/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.2446, 'grad_norm': 11.454587936401367, 'learning_rate': 6.666666666666667e-06, 'epoch': 1.0}        
{'eval_loss': 0.24014005064964294, 'eval_accuracy': 0.9128440366972477, 'eval_runtime': 2.9325, 'eval_samples_per_second': 297.359, 'eval_steps_per_second': 4.774, 'epoch': 1.0}                                             
 33%|███████████████████████▎                                              | 1053/3159 [08:40<14:32,  2.41it/s/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.1358, 'grad_norm': 2.009165048599243, 'learning_rate': 3.3333333333333333e-06, 'epoch': 2.0}        
{'eval_loss': 0.25326141715049744, 'eval_accuracy': 0.9151376146788991, 'eval_runtime': 2.9421, 'eval_samples_per_second': 296.387, 'eval_steps_per_second': 4.759, 'epoch': 2.0}                                             
 67%|██████████████████████████████████████████████▋                       | 2106/3159 [17:30<07:14,  2.42it/s/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
{'loss': 0.1027, 'grad_norm': 1.5390743017196655, 'learning_rate': 0.0, 'epoch': 3.0}                          
{'eval_loss': 0.26920342445373535, 'eval_accuracy': 0.9174311926605505, 'eval_runtime': 2.9235, 'eval_samples_per_second': 298.275, 'eval_steps_per_second': 4.789, 'epoch': 3.0}                                             
{'train_runtime': 1593.7594, 'train_samples_per_second': 126.774, 'train_steps_per_second': 1.982, 'train_loss': 0.1610608826939461, 'epoch': 3.0}                                                                            
100%|██████████████████████████████████████████████████████████████████████| 3159/3159 [26:33<00:00,  1.98it/s]
/home0/y2022/f2210543/.pyenv/versions/3.10.13/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
  warnings.warn('Was asked to gather along dimension 0, but all '
100%|██████████████████████████████████████████████████████████████████████████| 14/14 [00:02<00:00,  5.19it/s]
正解率: {'eval_loss': 0.26920342445373535, 'eval_accuracy': 0.9174311926605505, 'eval_runtime': 2.9571, 'eval_samples_per_second': 294.886, 'eval_steps_per_second': 4.734, 'epoch': 3.0}
"""