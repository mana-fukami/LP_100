#無関係の下位問題を抽出する
#短いコードを書く

# Ward法によるクラスタリング
# 国名に関する単語ベクトルに対し、Ward法による階層型クラスタリングを実行せよ。
# さらに、クラスタリング結果をデンドログラムとして可視化せよ。

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from WordVector import Word2Vec

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

linkage_matrix=linkage(country_vec,method="ward")

plt.figure(figsize=(10,7))
dendrogram(linkage_matrix,labels=country_name,leaf_rotation=90,leaf_font_size=10)
plt.title("Hierarchical Clustering with Ward Method")
plt.xlabel("Country")
plt.ylabel("Distance")
plt.show()

# 実行結果→Figure_1.png