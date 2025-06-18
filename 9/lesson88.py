"""
問題87でファインチューニングされたモデルを用いて、以下の文の極性を予測せよ。
    “The movie was full of incomprehensibilities.”
    “The movie was full of fun.”
    “The movie was full of excitement.”
    “The movie was full of crap.”
    “The movie was full of rubbish.”
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ファインチューニング済みのモデルの読み込み
model_dir="/home0/y2022/f2210543/LP_100/9/results/checkpoint-6315"
model=AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer=AutoTokenizer.from_pretrained(model_dir)

# 入力文
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# 文の極性予測
for sentence in sentences:
    inputs=tokenizer(sentence,return_tensors="pt",padding=True,truncation=True)
    outputs=model(**inputs)
    predictions=torch.argmax(outputs.logits,dim=1)
    print(f"sentence: {sentence} -> Prediction: {predictions.item()}")
"""
sentence: The movie was full of incomprehensibilities. -> Prediction: 1
sentence: The movie was full of fun. -> Prediction: 1
sentence: The movie was full of excitement. -> Prediction: 1
sentence: The movie was full of crap. -> Prediction: 1
sentence: The movie was full of rubbish. -> Prediction: 1
"""