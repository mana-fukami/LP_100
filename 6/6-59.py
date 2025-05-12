#無関係の下位問題を抽出する
#短いコードを書く

# t-SNEによる可視化
# ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ。

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from WordVector import Word2Vec
import numpy as np

w2v=Word2Vec()

file=open("6/country-names.txt","r",encoding="utf-8")
lines=file.readlines()

country=[]
for line in lines:
    country.append(line.replace("\n",""))

country_vec=[]
country_name=[]
vec=None
for name in country:
    vec=w2v.get_vector(name)
    if vec is not None:
        country_vec.append(vec)
        country_name.append(name)

tsne=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
reduced_vecs=tsne.fit_transform(np.array(country_vec))

plt.figure(figsize=(12,8))
for i,label in enumerate(country_name):
    x,y=reduced_vecs[i]
    plt.scatter(x, y)
    plt.text(x + 0.5, y + 0.5, label, fontsize=9)

plt.title("t-SNE Visualization of Country Vectors")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

# 実行結果→Figure_2.png