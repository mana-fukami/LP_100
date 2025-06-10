"""
“The movie was full of incomprehensibilities.”
という文をトークンに分解し、トークン列を表示せよ。
"""
from transformers import AutoTokenizer

# トークナイザーのロード
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

# テキストのトークン化
text="The movie was full of incomprehensibilities."
tokens=tokenizer.tokenize(text)
print(tokens)
# 実行結果
"""
['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']
"""