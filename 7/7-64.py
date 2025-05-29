"""
学習したロジスティック回帰モデルを用い、
検証データの先頭の事例を各ラベル（ポジネガ）に分類するときの条件付き確率を求めよ。
"""
import pandas as pd
from MachineLearn import DataSummary,LogisticLearning

# データの読み込み
train=open("SST-2/train.tsv","r")
dev=open("SST-2/dev.tsv","r")
#df=[sentence][label]
train_df=pd.read_csv(train,sep="\t")
dev_df=pd.read_csv(dev,sep="\t")
# モデルの学習
log_learn=LogisticLearning(train_df)
model=log_learn.learned_model()

# 条件付き確率
# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
dev_dict_list=DataSummary(dev_df).organize_data()
# データフレーム型に変換
train_data=pd.DataFrame(dev_dict_list)
# 予測用の入力値
x_pred=log_learn.vec.transform(train_data["feature"])
x_sample=x_pred[0]
# 条件付き確率を求める
probs=model.predict_proba([x_sample])

print(f"ネガティブの確率: {probs[0][0]:.4f}")
print(f"ポジティブの確率: {probs[0][1]:.4f}")

# 実行結果
"""
ネガティブの確率: 0.0040
ポジティブの確率: 0.9960
"""