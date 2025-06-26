"""
事前学習済み言語モデルで感情分析を行いたい。
テキストを含むプロンプトを事前学習済み言語モデルに与え、
(ファインチューニングは行わずに)テキストのポジネガを予測するという戦略で、
SST-2の開発データにおける正解率を測定せよ。
"""
import torch
from transformers import AutoTokenizer,GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm

# 事前学習済みモデルの読み込み
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
model=GPT2LMHeadModel.from_pretrained("openai-community/gpt2-medium",num_labels=2)
model.eval()

# GPUに移動
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# SSTの開発データの読み込み
dev_tsv=open("SST-2/dev.tsv","r")
dev_df=pd.read_csv(dev_tsv,sep="\t")

# 評価用関数
def predict_sentiment(text):
    prompt=f"Review: {text}\nSentiment:"
    inputs=tokenizer(prompt,return_tensors="pt").to(device)
    # 出力の最初のトークン予測
    with torch.no_grad():
        outputs=model(**inputs)
        next_token_logits=outputs.logits[0,-1,:]
        probs=torch.softmax(next_token_logits,dim=-1)
    pos_id=tokenizer.encode(" positive",add_special_tokens=False)[0]
    neg_id=tokenizer.encode(" negative",add_special_tokens=False)[0]
    return 1 if probs[pos_id]>probs[neg_id] else 0

# 予測と精度評価
correct=0
for _,row in tqdm(dev_df.iterrows(),total=len(dev_df)):
    pred=predict_sentiment(row["sentence"])
    if pred==row["label"]:
        correct+=1
accuracy=correct/len(dev_df)
print(f"正解率: {accuracy}")
"""
100%|██████████████████████| 872/872 [03:08<00:00,  4.64it/s]
正解率: 0.75
"""