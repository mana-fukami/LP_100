"""
“The movie was full of”に続くテキストを複数予測せよ。
このとき、デコーディングの方法や温度パラメータ（temperature）を変えながら、予測される複数のテキストの変化を観察せよ。
"""
from transformers import AutoTokenizer,AutoModelForCausalLM

# トークナイザーの読み込み
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token=tokenizer.eos_token # GPT2に必要
# モデルの読み込み
#   CausalLM：因果言語モデリングに特化したもの。主に文章の続きを予測するタスクに使われる。
model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
model.eval()

# プロンプトの設定
prompt="The movie was full of"
inputs=tokenizer(prompt,return_tensors="pt")
input_ids=inputs["input_ids"]

# デコード設定一覧
settings = [
    {"name": "greedy", "do_sample": False},
    {"name": "sampling_temp_1.0", "do_sample": True, "temperature": 1.0},
    {"name": "sampling_temp_0.7", "do_sample": True, "temperature": 0.7},
    {"name": "top_k_50", "do_sample": True, "top_k": 50},
    {"name": "top_p_0.9", "do_sample": True, "top_p": 0.9},
]

# 各設定で生成
for setting in settings:
    print(f"\n--- {setting['name']} ---")
    gen_kwargs = setting.copy()
    gen_kwargs.pop("name")  # modelに不要な'name'を除去
    output = model.generate(
        input_ids,
        max_length=20,
        **gen_kwargs,
        pad_token_id=tokenizer.eos_token_id
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))

# 実行結果
"""
--- greedy ---
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
The movie was full of great moments, but the most memorable was when the characters were reunited.


--- sampling_temp_1.0 ---
The movie was full of moments of triumph, with a hero's journey taking the center stage and an

--- sampling_temp_0.7 ---
The movie was full of great moments and great moments. I think that's the best way to think

--- top_k_50 ---
The movie was full of big moments, but my favorite was Tom Hanks's speech at the movie

--- top_p_0.9 ---
The movie was full of them, all those long-running character arcs, from the quirky, goofy
"""