"""
適当な文を準備して、事前学習済み言語モデルでパープレキシティを測定せよ。
例えば、
    The movie was full of surprises
    The movies were full of surprises
    The movie were full of surprises
    The movies was full of surprises
の4文に対して、パープレキシティを測定して観察せよ
(最後の2つの文は故意に文法的な間違いを入れた)。
"""
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import torch.nn.functional as F
import math

# トークナイザーの読み込み
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token=tokenizer.eos_token # GPT2に必要
# モデルの読み込み
#   CausalLM：因果言語モデリングに特化したもの。主に文章の続きを予測するタスクに使われる。
model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
model.eval()

sentences=[
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

def compute_perplexity(sentence):
    inputs=tokenizer(sentence,return_tensors="pt")
    input_ids=inputs["input_ids"]
    with torch.no_grad():
        outputs=model(input_ids,labels=input_ids)
            # 入力と正解を同じにして、自己回帰的に単語の予測損失を計算
        loss=outputs.loss
    ppl=math.exp(loss.item())
    return ppl

print("Perplexity Results:")
for sentence in sentences:
    ppl=compute_perplexity(sentence)
    print(f"{sentence:<40} PPL = {ppl:.2f}")

"""
Perplexity Results:
The movie was full of surprises          PPL = 89.45
The movies were full of surprises        PPL = 164.89
The movie were full of surprises         PPL = 324.10
The movies was full of surprises         PPL = 388.44
"""
# `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.