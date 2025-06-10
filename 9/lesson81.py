"""
“The movie was full of [MASK].”の”[MASK]”を埋めるのに最も適切なトークンを求めよ。
"""
from transformers import pipeline

# paielineのロード
fill_mask=pipeline("fill-mask",model="bert-base-uncased",top_k=1)

# テキストのトークン化
text="The movie was full of [MASK]."
print(fill_mask(text))
# 実行結果
"""
[{'score': 0.10711949318647385, 'token': 4569, 'token_str': 'fun', 'sequence': 'the movie was full of fun.'}]
"""