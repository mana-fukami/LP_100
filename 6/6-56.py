#無関係の下位問題を抽出する
#短いコードを書く

# WordSimilarity-353での評価
# The WordSimilarity-353 Test Collectionの評価データをダウンロードし、単語ベクトルにより計算される類似度のランキングと、
# 人間の類似度判定のランキングの間のスピアマン相関係数を計算せよ。

import pandas as pd
from WordVector import Word2Vec
from scipy.stats import spearmanr

## wordsimファイルの読み込み
df=pd.read_csv("wordsim353/combined.csv")
## 単語ベクトルの読み込み
w2v=Word2Vec()

## 類似度の計算
vec_sim=[]
human_sim=[]

for index,row in df.iterrows():
    w1,w2=row["Word 1"],row["Word 2"]
    sim=w2v.word_cos_sim(w1,w2)
    vec_sim.append(sim)
    human_sim.append(row["Human (mean)"])

# スピアマン相関係数を計算する
rho, p_value=spearmanr(vec_sim,human_sim)

print(f"スピアマンの相関係数: {rho}")

# 実行結果
#  スピアマンの相関係数: 0.7000166486272194