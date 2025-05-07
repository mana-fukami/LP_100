#制御フローを読みやすくする
#日本に関する記事における名詞のTF・IDFスコアを求め、
# TF・IDFスコア上位20語とそのTF, IDF, TF・IDFを表示せよ。

import json
from RemoveMarkup import remove_markup
from TokenAppearanceFrequency import token_class_appearance_frequency
from math import log

corpus=open("4\jawiki-country.json","r",encoding="utf-8")
corpus_lines=corpus.readlines()
#print(corpus_lines[0])

articles=[]
for line in corpus_lines:
    articles.append(json.loads(line))
#print(articles[0]["text"])

#Wikipediaマークアップの除去
for article in articles:
    article["text"]=remove_markup(article["text"])

#各名詞の出現回数
noun_frequency={} #{key:形態素の表層形,value:出現回数}
for article in articles:
    text=article["text"]
    noun_frequency=token_class_appearance_frequency(noun_frequency,text,"名詞")
#TFスコアの計算
sum_frequency=0
for keitaiso,value in noun_frequency.items():
    sum_frequency+=value
noun_tf={} #key:形態素,value:tfスコア
for keitaiso,value in noun_frequency.items():
    noun_tf[keitaiso]=value/sum_frequency

#各名詞の出現する文書の数
noun_df={}
for keitaiso,value in noun_frequency.items():
    for article in articles:
        text=article["text"]
        if keitaiso in text:
            if keitaiso not in noun_df:
                noun_df[keitaiso]=1
            else:
                noun_df[keitaiso]+=1
#名詞のIDFスコア：上位20語を表示
sum_document=len(articles)
noun_idf={}
for keitaiso,value in noun_df.items():
    noun_idf[keitaiso]=log(sum_document/value)+1

#TF・IDFスコア
noun_tf_idf={}
for keitaiso,value in noun_tf.items():
    noun_tf_idf[keitaiso]=value*(noun_idf.get(keitaiso,0))
#上位20語を表示
print("---tf-idfスコアの上位20語---")
print("形態素:tf-idfスコア:tfスコア:idfスコア")
top_20_tf_idf=sorted(noun_tf_idf.items(),key=lambda x:x[1],reverse=True)[:20]
for word,tf_idf_score in top_20_tf_idf:
    print(word+":"+str(tf_idf_score)+":"+str(noun_tf.get(keitaiso,0))+":"+str(noun_idf.get(keitaiso,0)))
print("---------------------------")

#実行結果
"""
---tf-idfスコアの上位20語---
形態素:tf-idfスコア:tfスコア:idfスコア
年:0.03167776474565873:3.001272539556772e-06:5.820281565605037
月:0.009774715337122057:3.001272539556772e-06:5.820281565605037
語:0.007546180051329478:3.001272539556772e-06:5.820281565605037
日本:0.006085648102664453:3.001272539556772e-06:5.820281565605037
こと:0.005565360581855986:3.001272539556772e-06:5.820281565605037
大統領:0.005432373790883395:3.001272539556772e-06:5.820281565605037
共和:0.005170637281429241:3.001272539556772e-06:5.820281565605037
1:0.005093554627441879:3.001272539556772e-06:5.820281565605037
世界:0.004970949102667174:3.001272539556772e-06:5.820281565605037
2:0.004444568902709902:3.001272539556772e-06:5.820281565605037
アメリカ:0.0043037400534787764:3.001272539556772e-06:5.820281565605037
ロシア:0.004113530663481147:3.001272539556772e-06:5.820281565605037
政府:0.004104235222056045:3.001272539556772e-06:5.820281565605037
フランス:0.004093014443854088:3.001272539556772e-06:5.820281565605037
3:0.004004347816702897:3.001272539556772e-06:5.820281565605037
ため:0.00398924998588038:3.001272539556772e-06:5.820281565605037
イギリス:0.003620286357383378:3.001272539556772e-06:5.820281565605037
経済:0.00348561494684961:3.001272539556772e-06:5.820281565605037
万:0.0033590474271362435:3.001272539556772e-06:5.820281565605037
主義:0.0032993492905863077:3.001272539556772e-06:5.820281565605037
---------------------------
"""