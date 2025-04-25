#コーパスにおける名詞の出現頻度を求め、
# 出現頻度の高い20語とその出現頻度を表示せよ。

import json
import MeCab
from RemoveMarkup import remove_markup

corpus=open("4\jawiki-country.json","r",encoding="utf-8")
corpus_lines=corpus.readlines()
#print(corpus_lines[0])

articles=[]
for line in corpus_lines:
    articles.append(json.loads(line))
#print(articles[0]["text"][1500:5000])

#Wikipediaマークアップの除去
for article in articles:
    article["text"]=remove_markup(article["text"])
#print(articles[0]["text"])

#形態素の出現頻度
keitaiso_surface_frequency={} #{key:形態素の表層形,value:出現回数}
for article in articles:
    text=article["text"]
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(text)
    while node:
        if node.surface != "":
            node_feature=node.feature.split(",")
            if node_feature[0] == "名詞":
                if node.surface not in keitaiso_surface_frequency:
                    keitaiso_surface_frequency[node.surface]=1
                else:
                    keitaiso_surface_frequency[node.surface]+=1
        node=node.next
#出現頻度でソート
sorted=sorted(keitaiso_surface_frequency.items(),key=lambda x:x[1],reverse=True)
#上位20個を取り出す
top_20=dict(sorted[:20])
for keitaiso,value in top_20.items():
    print(keitaiso+":"+str(value))

#実行結果
"""
年:20940
月:6358
語:4869
日本:3756
こと:3505
1:3367
世界:3031
2:2938
共和:2824
3:2647
大統領:2572
政府:2564
ため:2543
アメリカ:2293
経済:2213
4:2069
5:1992
独立:1979
フランス:1963
イギリス:1905
"""