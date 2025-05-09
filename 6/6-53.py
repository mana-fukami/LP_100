#無関係の下位問題を抽出する
#短いコードを書く

# 加法構成性によるアナロジー
# “Spain”の単語ベクトルから”Madrid”のベクトルを引き、”Athens”のベクトルを足したベクトルを計算し、
# そのベクトルと類似度の高い10語とその類似度を出力せよ。

from WordVector import Word2Vec
word2vec=Word2Vec()

word1="Spain"
word2="Madrid"
word3="Athens"

vec1=word2vec.get_vector(word1)
vec2=word2vec.get_vector(word2)
vec3=word2vec.get_vector(word3)

sub_vec=vec1-vec2
add_vec=sub_vec+vec3

sim_list=word2vec.vec_cos_sim_search(add_vec)
# sim_list=model.most_similar(positive=[word1,word3],negative[word2])
for i in range(10):
    sim=sim_list[i]
    print(f"{sim[0]}: {sim[1]}")

#実行結果
"""
Athens: 0.7528455853462219
Greece: 0.6685472130775452
Aristeidis_Grigoriadis: 0.5495778322219849
Ioannis_Drymonakos: 0.5361456871032715
Greeks: 0.5351787209510803
Ioannis_Christou: 0.5330225825309753
Hrysopiyi_Devetzi: 0.5088489055633545
Iraklion: 0.5059264898300171
Greek: 0.5040615797042847
Athens_Greece: 0.5034109354019165
"""