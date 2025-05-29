"""
61で構築した学習データの特徴ベクトルを用いて、ロジスティック回帰モデルを学習せよ。
"""
from MachineLearn import DataSummary
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

# データの読み込み
train=open("SST-2/train.tsv","r")
#df=[sentence][label]
train_df=pd.read_csv(train,sep="\t")
# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
train_dict_list=DataSummary(train_df).organize_data()
# データフレーム型に変換
train_data=pd.DataFrame(train_dict_list)

# 学習用の入力値と目標値
vec=DictVectorizer(sparse=False)
x_train=vec.fit_transform(train_data["feature"])
t_train=train_data["label"]

log_model=LogisticRegression()
log_model.fit(x_train,t_train)
print("---学習結果---")
print("パラメータ: {}".format(log_model.coef_))
print("バイアス: {}".format(log_model.intercept_))
print("精度の検証: {}".format(log_model.score(x_train,t_train)))

# 実行結果
"""
---学習結果---
パラメータ: [[ 0.84901029  0.33538168 -0.59808355 ...  0.07756328 -1.76994755
  -0.04840973]]
バイアス: [0.33491186]
精度の検証: 0.9414839121590521
"""




