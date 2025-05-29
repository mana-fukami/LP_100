"""
与えられたテキストのポジネガを予測するプログラムを実装せよ。
例えば、テキストとして”the worst movie I ‘ve ever seen”を与え、
ロジスティック回帰モデルの予測結果を確認せよ。
"""
import pandas as pd
from MachineLearn import LogisticLearning

# データの読み込み
train=open("SST-2/train.tsv","r")
dev=open("SST-2/dev.tsv","r")
#df=[sentence][label]
train_df=pd.read_csv(train,sep="\t")
dev_df=pd.read_csv(dev,sep="\t")
# モデルの学習
log_learn=LogisticLearning(train_df)
model=log_learn.learned_model()

def get_feature_dict(sentence):
    feature={}
    splitted=sentence.split(" ")
    for word in splitted:
        if word!="":
            if word not in feature:
                feature[word]=1
            else:
                feature[word]+=1
    return feature

text="the worst movie I ‘ve ever seen"
x_pred=log_learn.vec.transform(get_feature_dict(text))
predict=log_learn.model.predict(x_pred)

print("予測結果: {}".format(predict))

# 実行結果
"""
予測結果: [0]
"""