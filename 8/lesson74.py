"""
問題73で学習したモデルの開発セットにおける正解率を求めよ。
"""
from lesson73 import model
from lesson72 import dev_dataloader,device
import torch

model.eval()
correct=0
total=0

with torch.no_grad(): #計算の高速化のために勾配計算を無効にする
    for batch in dev_dataloader:
        features=batch["feature_vec"].to(device)
        labels=batch["label"].to(device).view(-1,1)

        # モデルの出力を取得
        outputs=model(features)

        # 0.5を閾値として二値分類
        predicted=(outputs>0.5).float()

        #正解数のカウント
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

# 正解率を計算
accuracy=100*correct/total
print(f"正解率: {accuracy:.2f}%")
# 正解率: 79.82%