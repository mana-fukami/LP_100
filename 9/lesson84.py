"""
以下の文の全ての組み合わせに対して、最終層の埋め込みベクトルの平均を用いてコサイン類似度を求めよ。
    “The movie was full of fun.”
    “The movie was full of excitement.”
    “The movie was full of crap.”
    “The movie was full of rubbish.”
"""
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# トークナイザーとモデルのロード
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
model=AutoModel.from_pretrained("bert-base-uncased")

# テキストの定義
text1="The movie was full of fun."
text2="The movie was full of excitement."
text3="The movie was full of crap."
text4="The movie was full of rubbish."
texts=[text1, text2, text3, text4]

# 各テキストのトークン埋め込みベクトルを取得
cls_embeddings=[]
for text in texts:
    # PyTorchのテンソルに形式でトークン化
    inputs=tokenizer(text,return_tensors="pt",padding=True, truncation=True)
    with torch.no_grad():
        outputs=model(**inputs)
    # 最終層の埋め込みベクトルを取得
    last_hidden_state_vec=outputs.last_hidden_state[0]
    # 埋め込みベクトルの平均を計算
    avg_vec=torch.mean(last_hidden_state_vec, dim=0)
    # 平均ベクトルをリストに追加
    cls_embeddings.append(avg_vec)

# コサイン類似度の計算
cosine_sim=[]
for vec1 in cls_embeddings:
    sim=[]
    #print(vec1.shape)
    for vec2 in cls_embeddings:
        sim.append(F.cosine_similarity(vec1,vec2,dim=0))
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
    0|1.0000 0.9568 0.8490 0.8169
    1|0.9568 1.0000 0.8352 0.7938
    2|0.8490 0.8352 1.0000 0.9226
    3|0.8169 0.7938 0.9226 1.0000
"""