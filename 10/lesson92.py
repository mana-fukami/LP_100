"""
“The movie was full of”に続くテキストを予測し、生成された各単語の尤度を表示せよ
(生成されるテキストが長いと出力が読みにくくなるので、適当な長さで生成を打ち切るとよい）。
"""
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import torch.nn.functional as F

# トークナイザーの読み込み
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token=tokenizer.eos_token # GPT2に必要
# モデルの読み込み
#   CausalLM：因果言語モデリングに特化したもの。主に文章の続きを予測するタスクに使われる。
model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
model.eval()

# 入力文
prompt="The movie was full of"
inputs=tokenizer(prompt,return_tensors="pt",padding=True)
input_ids=inputs["input_ids"]
attention_mask=inputs["attention_mask"]

# 生成：次のトークンを5つ生成（temperature=1.0）
with torch.no_grad():
    output_ids=model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False, # 最も尤度の高いトークンを逐次選択
        return_dict_in_generate=True,
        output_scores=True, # 各トークンのlogitsスコアを出力
        pad_token_id=tokenizer.eos_token_id
    )

# 生成結果
generated_ids=output_ids.sequences[0]
generated_text=tokenizer.decode(generated_ids,skip_special_tokens=True)
print(f"Generated Text:\n{generated_text}")

# 出力スコアから尤度を計算
print("Token-wise Probabilities:")
for i in range(0,len(output_ids.scores)):
    token_id=generated_ids[i+5].item() # 入力トークン数は5なので
    token_str=tokenizer.decode([token_id])
    score=output_ids.scores[i] # output_idsには生成結果のみなので、0番目から参照する(i-1)
    probs=F.softmax(score,dim=-1) # logitsを正規化→確率
    log_probs=F.log_softmax(score,dim=-1) # logitsの対数確率
    token_prob=probs[0,token_id].item() # 該当トークンの出現確率
    token_logprob=log_probs[0,token_id].item() # その対数確率
    print(f"{i:>2}: {token_str:<10}\tProb: {token_prob:.5f}\tLogProb: {token_logprob:.5f}")

"""
Generated Text:
The movie was full of great moments, but the
Token-wise Probabilities:
 0:  great      Prob: 0.02309   LogProb: -3.76817
 1:  moments    Prob: 0.18811   LogProb: -1.67073
 2: ,           Prob: 0.36348   LogProb: -1.01203
 3:  but        Prob: 0.22248   LogProb: -1.50290
 4:  the        Prob: 0.13663   LogProb: -1.99044
"""