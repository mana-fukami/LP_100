"""
訓練セットを用い、事前学習済みモデルを極性分析タスク向けにファインチューニングせよ。
検証セット上でファインチューニングされたモデルの正解率を計測せよ。
"""
from lesson85 import train_dataset,dev_dataset
from lesson86 import CustomCollate
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

# 事前学習済みモデルの読み込み
model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
# データローダーの作成
train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True,collate_fn=CustomCollate)
dev_dataloader=DataLoader(dev_dataset,batch_size=64,shuffle=True,collate_fn=CustomCollate)
# モデルをGPUに移動（必要に応じて）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model=nn.DataParallel(model)
model.to(device)
# 最適化関数の設定
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn=CrossEntropyLoss()

# ファインチューニングのループ
model.train()
num_epochs=10
best_valid_loss=float("inf") # lossは小さくしていきたいから初期設定はinfを設定
best_valid_accuracy=0 # correctは大きくしていきたいから初期値は0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 入力データをデバイスに移動
        inputs=batch["tokens"]["input_ids"].to(device)
        attention_mask=batch["tokens"]["attention_mask"].to(device)
        labels=batch["label"].to(device)
        # モデルの出力を計算
        outputs=model(inputs,attention_mask=attention_mask)
        logits=outputs.logits
        # 損失を計算
        loss=loss_fn(logits,labels)
        # 勾配を計算して更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 検証セットで評価
    model.eval()
    total_loss=0
    correct=0
    total=0
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs=batch["tokens"]["input_ids"].to(device)
            attention_mask=batch["tokens"]["attention_mask"].to(device)
            labels=batch["label"].to(device)
            # モデルの出力を取得
            outputs=model(inputs,attention_mask=attention_mask)
            logits=outputs.logits
            # 損失を計算
            loss=loss_fn(outputs,labels)
            total_loss+=loss.item()
            #予測を取得
            predicted=torch.argmax(logits,dim=1)
            # 正解数のカウント
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        # 平均損失と正解率を計算
        avg_loss=total_loss/len(dev_dataloader)
        accuracy=100*correct/total
    if best_valid_loss>avg_loss:
        best_valid_loss=avg_loss
    if best_valid_accuracy<accuracy:
        best_valid_accuracy=accuracy
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
# 最良のスコアを出力
print(f"Best Valid Loss: {best_valid_loss:.4f}, Best Valid Accuracy: {best_valid_accuracy:.2f}%")