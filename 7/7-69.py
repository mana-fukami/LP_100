"""
ロジスティック回帰モデルを学習するとき、正則化の係数（ハイパーパラメータ）を調整することで、
学習時の適合度合いを制御できる。正則化の係数を変化させながらロジスティック回帰モデルを学習し、
検証データ上の正解率を求めよ。実験の結果は、正則化パラメータを横軸、正解率を縦軸としたグラフにまとめよ。
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from MachineLearn import DataSummary

#-----データの読み込み-----
train=open("SST-2/train.tsv","r")
dev=open("SST-2/dev.tsv","r")
train_df=pd.read_csv(train,sep="\t")
dev_df=pd.read_csv(dev,sep="\t")

#-----データを適切な形式に整形-----
# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
train_dict_list=DataSummary(train_df).organize_data()
dev_dict_list=DataSummary(dev_df).organize_data()
# リストをデータフレーム型に変換
train_data=pd.DataFrame(train_dict_list)
dev_data=pd.DataFrame(dev_dict_list)

#-----データの分割-----
vec=DictVectorizer(sparse=False)
# 学習データ
x_train=vec.fit_transform(train_data["feature"])
t_train=train_data["label"]
# 検証データ
x_dev=vec.transform(dev_data["feature"])
t_dev=dev_data["label"]

#-----学習と正解率-----
# 正則化パラメータの候補
C_list=[0.01, 0.1, 1, 10, 100]
# 正解率
accuracies=[]
for C in C_list:
    model=LogisticRegression(C=C)
    model.fit(x_train,t_train)
    pred=model.predict(x_dev)
    acc=accuracy_score(t_dev,pred)
    accuracies.append(acc)

#-----グラフの描画-----
plt.figure(figsize=(8,5))
plt.semilogx(C_list,accuracies,marker='o')
plt.xlabel("C paremeter")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

#実行結果 → Figura1.png