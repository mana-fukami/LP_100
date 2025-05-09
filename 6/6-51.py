# 単語の類似度
# “United States”と”U.S.”のコサイン類似度を計算せよ。

from WordVector import Word2Vec
import numpy as np

word_vec=Word2Vec()

vec1=word_vec.get_vector("United States")
vec2=word_vec.get_vector("U.S.")

dot_product=np.dot(vec1,vec2)
norm_vec1=np.linalg.norm(vec1)
norm_vec2=np.linalg.norm(vec2)
cosine_sim=dot_product/(norm_vec1*norm_vec2)
#cosine_sim=model.similarity("United_States","U.S.")

print(f"cosine similarity: {cosine_sim}")

#実行結果
"""
cosine similarity: 0.7310774922370911
"""