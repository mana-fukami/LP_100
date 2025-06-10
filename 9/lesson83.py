"""
以下の文の全ての組み合わせに対して、最終層の[CLS]トークンの埋め込みベクトルを用いてコサイン類似度を求めよ。
    “The movie was full of fun.”
    “The movie was full of excitement.”
    “The movie was full of crap.”
    “The movie was full of rubbish.”
"""
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
model=AutoModel.from_pretrained("bert-base-uncased")

text1="The movie was full of fun."
text2="The movie was full of excitement."
text3="The movie was full of crap."
text4="The movie was full of rubbish."
texts=[text1, text2, text3, text4]

cls_embeddings=[]
for text in texts:
    # PyTorchのテンソルに形式でトークン化
    inputs=tokenizer(text,return_tensors="pt")
    with torch.no_grad():
        outputs=model(**inputs)
    # 最終層の[CLS]トークンの埋め込みベクトルを取得
    last_hidden_state=outputs.last_hidden_state[:,0,:]
    cls_embeddings.append(last_hidden_state)

cosine_sim=[]
for vec1 in cls_embeddings:
    sim=[]
    for vec2 in cls_embeddings:
        sim.append(F.cosine_similarity(vec1,vec2))
    cosine_sim.append(sim)

print("text1:",text1)
print("text2:",text2)
print("text3:",text3)
print("text4:",text4)
print("Cosine Similarity Matrix:")
print("     |1     2     3     4")
print("-----|-------------------")
for i in range(len(cosine_sim)):
    print(f"    {i}|", end="")
    for col in cosine_sim[i]:
        print(f"{col.item():.4f} ", end="")
    print()
# 実行結果
"""
text1: The movie was full of fun.
text2: The movie was full of excitement.
text3: The movie was full of crap.
text4: The movie was full of rubbish.
Cosine Similarity Matrix:
     |1     2     3     4
-----|-------------------
    0|1.0000 0.9881 0.9558 0.9475
    1|0.9881 1.0000 0.9541 0.9487
    2|0.9558 0.9541 1.0000 0.9807
    3|0.9475 0.9487 0.9807 1.0000
"""