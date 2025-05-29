"""
GLUEのウェブサイトからSST-2データセットを取得せよ。
学習データ（train.tsv）と検証データ（dev.tsv）のぞれぞれについて、ポジティブ (1) とネガティブ (0) の事例数をカウントせよ。
"""
import pandas as pd

dev=open("SST-2/dev.tsv","r")
train=open("SST-2/train.tsv","r")

#df=[sentence][label]
dev_df=pd.read_csv(dev,sep="\t")
train_df=pd.read_csv(train,sep="\t")

def count_pos_neg(df):
    pos=0
    neg=0
    for i in range(df.shape[0]):
        if df.loc[i,"label"]==1:
            pos+=1
        else:
            neg+=1
    return pos,neg

dev_pos,dev_neg=count_pos_neg(dev_df)
train_pos,train_neg=count_pos_neg(train_df)
print("---dev---\npos: {}\nneg: {}".format(dev_pos,dev_neg))
print("--train--\npos: {}\nneg: {}".format(train_pos,train_neg))
# 実行結果
"""
---dev---
pos: 444
neg: 428
--train--
pos: 37569
neg: 29780
"""