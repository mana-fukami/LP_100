"""
問題72で設計したモデルの重みベクトルを訓練セット上で学習せよ。
ただし、学習中は単語埋め込み行列の値を固定せよ（単語埋め込み行列のファインチューニングは行わない）。
また、学習時に損失値を表示するなど、学習の進捗状況をモニタリングできるようにせよ。
"""
from lesson72 import train_dataloader,model,device
import torch
from torch import nn

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

# 実行結果
"""
Using cpu device
Epoch 1/10, Loss: 0.34913399815559387
Epoch 2/10, Loss: 0.388567715883255
Epoch 3/10, Loss: 0.36910033226013184
Epoch 4/10, Loss: 0.32375213503837585
Epoch 5/10, Loss: 0.40922683477401733
Epoch 6/10, Loss: 0.2908661365509033
Epoch 7/10, Loss: 0.2154208868741989
Epoch 8/10, Loss: 0.21733129024505615
Epoch 9/10, Loss: 0.8838637471199036
Epoch 10/10, Loss: 0.2520342171192169
"""