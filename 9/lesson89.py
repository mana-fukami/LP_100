"""
問題87とは異なるアーキテクチャ（例えば[CLS]トークンを用いるか、各トークンの最大値プーリングを用いるなど）の分類モデルを設計し、
事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。検証セット上でファインチューニングされたモデルの正解率を計測せよ。
"""
from lesson85 import train_dataset,dev_dataset,tokenizer
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,Trainer,TrainingArguments,EarlyStoppingCallback
from sklearn.metrics import accuracy_score

# GPUに移動
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# 事前学習済みモデルの読み込み
model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2).to(device)
model=nn.DataParallel(model)
model=model.to(device)

# Trainerの設定
training_args=TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    label_names=["labels"],
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

# コラート関数
def CustomCollate(batch):
    #データを各項目についてまとめる
    labels=torch.tensor([item["label"] for item in batch])
    texts=[item["text"] for item in batch]
    tokens=[item["tokens"] for item in batch]
    #テキストをパディングありでトークナイズする
    padded_tokens=tokenizer(texts,padding=True,return_tensors="pt")

    return {"labels":labels,"input_ids":padded_tokens["input_ids"],"attention_mask":padded_tokens["attention_mask"],"token_type_ids":padded_tokens["token_type_ids"]}

# 評価指標
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    preds=torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy": accuracy_score(labels,preds)}

# Trainerで学習
trainer=Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=CustomCollate,
    compute_metrics=compute_metrics,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train(ignore_keys_for_eval=['last_hidden_state', 'hidden_states', 'attentions'])

# モデルの評価
model.eval()
correct=0
total=0

dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True, collate_fn=CustomCollate)
with torch.no_grad(): #計算の高速化のために勾配計算を無効にする
    for batch in dev_dataloader:
        features=batch["input_ids"]
        labels=batch["labels"].to(device).view(-1,1)

        # モデルの出力を取得
        outputs=model(features, attention_mask=batch["attention_mask"].to(device))
        logits = outputs.logits

        # 0.5を閾値として二値分類
        preds = torch.argmax(logits, dim=1)

# 正解率を計算
print(f"正解率: {accuracy_score(labels,preds)}%")