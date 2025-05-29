"""
Bag of Words (BoW) に基づき、学習データ（train.tsv）および検証データ（dev.tsv）のテキストを特徴ベクトルに変換したい。
ここで、ある事例のテキストの特徴ベクトルは、テキスト中に含まれる単語（スペース区切りのトークン）の出現頻度で構成する。
例えば、”too loud , too goofy”というテキストに対応する特徴ベクトルは、以下のような辞書オブジェクトで表現される。

{'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
各事例はテキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトでまとめておく。
例えば、先ほどの”too loud , too goofy”に対してラベル”0”（ネガティブ）が付与された事例は、以下のオブジェクトで表現される。

{'text': 'too loud , too goofy', 'label': '0', 'feature': {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}}
学習データと検証データの各事例を上記のような辞書オブジェクトに変換したうえで、学習データと検証データのそれぞれを、
辞書オブジェクトのリストとして表現せよ。さらに、学習データの最初の事例について、正しく特徴ベクトルに変換できたか、目視で確認せよ。
"""

import pandas as pd

dev=open("SST-2/dev.tsv","r")
train=open("SST-2/train.tsv","r")

#df=[sentence][label]
dev_df=pd.read_csv(dev,sep="\t")
train_df=pd.read_csv(train,sep="\t")

# 特徴ベクトルに変換する
def feature_dict(sentence):
    feature={}
    splitted=sentence.split(" ")
    for word in splitted:
        if word!="":
            if word not in feature:
                feature[word]=1
            else:
                feature[word]+=1
    return feature

# テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトにまとめる
def organize_data(df):
    dict_list=[]
    for i in range(df.shape[0]):
        organized={}
        organized["text"]=df.loc[i,"sentence"]
        organized["label"]=df.loc[i,"label"]
        organized["feature"]=feature_dict(df.loc[i,"sentence"])
        dict_list.append(organized)
    return dict_list

organized_dev=organize_data(dev_df)
organized_train=organize_data(train_df)
print("-----dev-----")
for i in range(3):
    print(organized_dev[i])
print("-----train-----")
for i in range(3):
    print(organized_train[i])

# 実行例
"""
-----dev-----
-----dev-----
{'text': "it 's a charming and often affecting journey . ", 'label': 1, 'feature': {'it': 1, "'s": 1, 'a': 1, 'charming': 1, 'and': 1, 'often': 1, 'affecting': 1, 'journey': 1, '.': 1}}
{'text': 'unflinchingly bleak and desperate ', 'label': 0, 'feature': {'unflinchingly': 1, 'bleak': 1, 'and': 1, 'desperate': 1}}
{'text': 'allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . ', 'label': 1, 'feature': {'allows': 1, 'us': 1, 'to': 2, 'hope': 1, 'that': 1, 'nolan': 1, 'is': 1, 'poised': 1, 'embark': 1, 'a': 2, 'major': 1, 'career': 1, 'as': 1, 'commercial': 1, 'yet': 1, 'inventive': 1, 'filmmaker': 1, '.': 1}}
-----train-----
{'text': 'hide new secretions from the parental units ', 'label': 0, 'feature': {'hide': 1, 'new': 1, 'secretions': 1, 'from': 1, 'the': 1, 'parental': 1, 'units': 1}}
{'text': 'contains no wit , only labored gags ', 'label': 0, 'feature': {'contains': 1, 'no': 1, 'wit': 1, ',': 1, 'only': 1, 'labored': 1, 'gags': 1}}
{'text': 'that loves its characters and communicates something rather beautiful about human nature ', 'label': 1, 'feature': {'that': 1, 'loves': 1, 'its': 1, 'characters': 1, 'and': 1, 'communicates': 1, 'something': 1, 'rather': 1, 'beautiful': 1, 'about': 1, 'human': 1, 'nature': 1}}
"""