"""
“The movie was full of [MASK].”の”[MASK]”を埋めるのに最も適切なトークンを求めよ。
"""
from transformers import pipeline

# paielineのロード
fill_mask=pipeline("fill-mask",model="bert-base-uncased")

# テキストのトークン化
text="The movie was full of [MASK]."
print(fill_mask(text))
# 実行結果
"""
[{'score': 0.10711949318647385, 'token': 4569, 'token_str': 'fun', 'sequence': 'the movie was full of fun.'}, {'score': 0.06634482741355896, 'token': 20096, 'token_str': 'surprises', 'sequence': 'the movie was full of surprises.'}, {'score': 0.04468420892953873, 'token': 3689, 'token_str': 'drama', 'sequence': 'the movie was full of drama.'}, {'score': 0.027217086404561996, 'token': 3340, 'token_str': 'stars', 'sequence': 'the movie was full of stars.'}, {'score': 0.02541288174688816, 'token': 11680, 'token_str': 'laughs', 'sequence': 'the movie was full of laughs.'}]
"""
# 見やすく整形したもの
"""
[
{'score': 0.10711949318647385, 'token': 4569, 'token_str': 'fun', 'sequence': 'the movie was full of fun.'},
{'score': 0.06634482741355896, 'token': 20096, 'token_str': 'surprises', 'sequence': 'the movie was full of surprises.'},
{'score': 0.04468420892953873, 'token': 3689, 'token_str': 'drama', 'sequence': 'the movie was full of drama.'},
{'score': 0.027217086404561996, 'token': 3340, 'token_str': 'stars', 'sequence': 'the movie was full of stars.'},
{'score': 0.02541288174688816, 'token': 11680, 'token_str': 'laughs', 'sequence': 'the movie was full of laughs.'}
]
"""