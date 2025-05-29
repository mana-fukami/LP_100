"""
学習したロジスティック回帰モデルの正解率、適合率、再現率、F1スコアを、学習データおよび検証データ上で計測せよ。
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from MachineLearn import DataSummary,LogisticLearning

# データの読み込み
train=open("SST-2/train.tsv","r")
dev=open("SST-2/dev.tsv","r")
#df=[sentence][label]
train_df=pd.read_csv(train,sep="\t")
dev_df=pd.read_csv(dev,sep="\t")
# モデルの学習
log_learn=LogisticLearning(train_df)
model=log_learn.model

# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
train_dict_list=DataSummary(train_df).organize_data()
dev_dict_list=DataSummary(dev_df).organize_data()
# データフレーム型に変換
train_data=pd.DataFrame(train_dict_list)
dev_data=pd.DataFrame(dev_dict_list)
# 予測用の入力値と目標値
x_train=log_learn.vec.transform(train_data["feature"])
t_train=train_data["label"]
x_dev=log_learn.vec.transform(dev_data["feature"])
t_dev=dev_data["label"]
# 予測値の取得
y_train_pred=model.predict(x_train)
y_dev_pred=model.predict(x_dev)

# 学習データでの評価
print("-----【学習データ】-----")
print("正解率:", accuracy_score(t_train, y_train_pred))
print("適合率:", precision_score(t_train, y_train_pred))
print("再現率:", recall_score(t_train, y_train_pred))
print("F1スコア:", f1_score(t_train, y_train_pred))

# 検証データでの評価
print("-----【検証データ】-----")
print("正解率:", accuracy_score(t_dev, y_dev_pred))
print("適合率:", precision_score(t_dev, y_dev_pred))
print("再現率:", recall_score(t_dev, y_dev_pred))
print("F1スコア:", f1_score(t_dev, y_dev_pred))

# 実行結果
"""
-----【学習データ】-----
正解率: 0.9414839121590521
適合率: 0.94186902133922
再現率: 0.9539780137879634
F1スコア: 0.9478848468018143
-----【検証データ】-----
正解率: 0.8119266055045872
適合率: 0.8004291845493562
再現率: 0.8400900900900901
F1スコア: 0.8197802197802198
"""