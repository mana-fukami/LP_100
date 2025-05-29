"""
学習したロジスティック回帰モデルを用い、検証データの先頭の事例のラベル（ポジネガ）を予測せよ。
また、予測されたラベルが検証データで付与されていたラベルと一致しているか、確認せよ。
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

# 予測
# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
dev_dict_list=DataSummary(dev_df).organize_data()
# データフレーム型に変換
train_data=pd.DataFrame(dev_dict_list)
# 予測用の入力値と目標値
x_pred=log_learn.vec.transform(train_data["feature"])
t_pred=train_data["label"]

print("予測値")
print(model.predict(x_pred[:1]))
print(t_pred[0])

# 実行結果
"""
予測値
[1]
1
"""