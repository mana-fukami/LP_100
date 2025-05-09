#無関係の下位問題を抽出する
#短いコードを書く

# 類似度の高い単語10件
# “United States”とコサイン類似度が高い10語と、その類似度を出力せよ。

from WordVector import Word2Vec

word2vec=Word2Vec()

sim_list=[]

for word in word2vec.model.key_to_index:
    sim=word2vec.word_cos_sim("United States",word)
    sim_list.append((word,sim))

sim_list.sort(key=lambda x:x[1],reverse=True)
# sim_list=model.most_similar(United_States)
for i in range(10):
    sim=sim_list[i]
    print(f"{sim[0],sim[1]}")

#実行結果
"""
('United_States', 1.0)
('Unites_States', 0.7877249)
('Untied_States', 0.754137)
('United_Sates', 0.7400725)
('U.S.', 0.7310775)
('theUnited_States', 0.6404394)
('America', 0.61784106)
('UnitedStates', 0.61673117)
('Europe', 0.6132989)
('countries', 0.60448045)
"""