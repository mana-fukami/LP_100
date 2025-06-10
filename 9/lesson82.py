"""
“The movie was full of [MASK].”の”[MASK]”に埋めるのに適切なトークン上位10個と、その確率（尤度）を求めよ。
"""
from transformers import AutoTokenizer,AutoModelForMaskedLM,pipeline

# 予測のための準備
tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
model=AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
fill_mask=pipeline("fill-mask",model=model,tokenizer=tokenizer,top_k=10)

# 予測
text="The movie was full of [MASK]."
prediction=fill_mask(text)

for pred in prediction:
    token=pred["token_str"]
    score=pred["score"]
    print(f"{token}: {score}")

# 実行結果
"""
fun: 0.10711949318647385
surprises: 0.06634482741355896
drama: 0.04468420892953873
stars: 0.027217086404561996
laughs: 0.02541288174688816
action: 0.019516952335834503
excitement: 0.01903809793293476
people: 0.01829025335609913
tension: 0.015030616894364357
music: 0.014646267518401146
"""