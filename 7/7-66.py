"""
学習したロジスティック回帰モデルの検証データにおける混同行列（confusion matrix）を求めよ。
"""
import pandas as pd
from sklearn.metrics import confusion_matrix
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

# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
dev_dict_list=DataSummary(dev_df).organize_data()
# データフレーム型に変換
train_data=pd.DataFrame(dev_dict_list)
# 予測用の入力値と目標値
x_pred=log_learn.vec.transform(train_data["feature"])
t_pred=train_data["label"]

# 予測
y_pred=model.predict(x_pred)
# 混同行列を求める
cm=confusion_matrix(t_pred,y_pred,labels=model.classes_)
# 表示
print("混同行列:\n",cm)

# 実行結果
"""
混同行列:
 [[335  93]
 [ 71 373]]
"""