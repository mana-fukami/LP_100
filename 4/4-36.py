#問題36から39までは、Wikipediaの記事を以下のフォーマットで書き出したファイル
# jawiki-country.json.gzをコーパスと見なし、統計的な分析を行う。
# /1行に1記事の情報がJSON形式で格納される
# /各行には記事名が”title”キーに、記事本文が”text”キーの辞書オブジェクトに格納され、
#   そのオブジェクトがJSON形式で書き出される
# /ファイル全体はgzipで圧縮される
# まず、第3章の処理内容を参考に、Wikipedia記事からマークアップを除去し、
# 各記事のテキストを抽出せよ。そして、コーパスにおける単語（形態素）
# の出現頻度を求め、出現頻度の高い20語とその出現頻度を表示せよ。

import json
import re
import MeCab

corpus=open("4\jawiki-country.json","r",encoding="utf-8")
corpus_lines=corpus.readlines()
#print(corpus_lines[0])

articles=[]
for line in corpus_lines:
    articles.append(json.loads(line))
#print(articles[0]["text"][1500:5000])

#Wikipediaマークアップの除去
for article in articles:
    text=article["text"]
    #強調マークアップの除去
    text=re.sub("'*?","",text)
    #内部リンクの除去,表示文字は残す
    text=re.sub("\[\[([^\[]+)\|([^\[]+)\|","",text)
    text=re.sub("\[\[([^\[]+)\|","",text)
    text=re.sub("\[\[","",text)
    text=re.sub("\]\]","",text)
    #ファイルのマークアップの除去,説明文は残す
    text=re.sub("ファイル.*?\|.*?\|","",text)
    #外部リンクの除去
    text=re.sub("\[http.*?\]","",text)
    #カテゴリの除去
    text=re.sub("Category","",text)
    #リダイレクトの除去,記事名と説明は残す
    text=re.sub("\#REDIRECT","",text)
    #Cite関連の除去
    text=re.sub("\{\{Cite web .+\}\}","",text)
    text=re.sub("\{\{Cite journal .+\}\}","",text)
    text=re.sub("\{\{Cite book .+\}\}","",text)
    #テンプレートの除去
    text=re.sub("\{\{.*?\|.*?\|","",text)
    text=re.sub("\{\{.*?\|","",text)
    text=re.sub("\{\{","",text)
    text=re.sub("\}\}","",text)
    #<ref>~</ref>の除去
    text=re.sub("\<ref.*?\>.*</ref>","",text)
    #コメントアウトなど<~>の除去
    text=re.sub("\<.*?\>","",text)
    #そのほか記号の除去
    text=re.sub("=*?","",text)
    text=re.sub("\|","",text)
    text=re.sub("\;|\:","",text)
    text=re.sub("\*|\#","",text)
    article["text"]=text
#print(articles[0]["text"])

#形態素の出現頻度
keitaiso_surface_frequency={} #{key:形態素の表層形,value:出現回数}
for article in articles:
    text=article["text"]
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(text)
    while node:
        if node.surface != "":
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
の:72742
、:69781
に:50593
。:42524
は:42105
が:35282
を:30325
て:28420
た:28360
で:27504
と:27289
し:24103
年:20940
・:14125
（:13980
）:13753
れ:10935
いる:10090
さ:9948
ある:9298
"""