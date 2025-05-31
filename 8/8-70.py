"""
事前学習済み単語埋め込みを活用し、単語埋め込み行列を作成せよ。
また、単語埋め込み行列の構築と同時に、単語埋め込み行列の各行のインデックス番号(トークンID)と、
単語（トークン）への双方向の対応付けを保持せよ。
"""
import numpy as np
from gensim.models import KeyedVectors
model_path="GoogleNews-vectors-negative300.bin"
model=KeyedVectors.load_word2vec_format(model_path,binary=True)

id_to_token={}
token_to_id={}
embedding_matrix=[]
embedding_matrix.append(np.zeros(300))
for id,word in enumerate(model.index_to_key):
    id_to_token[id]=word
    token_to_id[word]=id
    embedding_matrix.append(model[word])

embedding_matrix = np.array(embedding_matrix)

print(embedding_matrix.shape)
print(id_to_token[100])
print(token_to_id[id_to_token[100]])

# 実行結果
"""
(3000001, 300)
company
100
"""