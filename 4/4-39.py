#コーパスにおける単語の出現頻度順位を横軸、その出現頻度を縦軸として、
# 両対数グラフをプロットせよ。

import json
import matplotlib.pyplot as plt
from RemoveMarkup import remove_markup
from TokenAppearanceFrequency import token_appearance_frequency

corpus=open("4\jawiki-country.json","r",encoding="utf-8")
corpus_lines=corpus.readlines()

articles=[]
for line in corpus_lines:
    articles.append(json.loads(line))

#Wikipediaマークアップの除去
for article in articles:
    article["text"]=remove_markup(article["text"])

#単語の出現頻度
token_frequency={}
for article in articles:
    token_frequency=token_appearance_frequency(token_frequency,article["text"])
frequency=[]
for keiteiso,value in token_frequency.items():
    frequency.append(value)
sorted_frequency=sorted(frequency,reverse=True)
rank=range(1,len(sorted_frequency)+1)

#両対数グラフをプロットする
plt.figure(figsize=(10,6))
plt.loglog(rank,sorted_frequency,marker="o",linestyle="none")
plt.title("both log graph")
plt.xlabel("frequency-ranking")
plt.ylabel("frequency")
plt.grid(True,which="both",linestyle="--",linewidth=0.5)
plt.show()