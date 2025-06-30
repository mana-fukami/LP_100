"""
問題96のプロンプトに対して、正解の感情ラベルをテキストの応答
として返すように事前学習済みモデルをファインチューニングせよ。
"""
import torch
from transformers import AutoTokenizer,GPT2Model,Trainer,TrainingArguments
import pandas as pd
from datasets import Dataset

# GPUに移動
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ読み込み
train_df=pd.read_csv("SST-2/train.tsv",sep="\t")
dev_df=pd.read_csv("SST-2/dev.tsv",sep="\t")

# モデルとトークナイザー
model=GPT2Model("openai-community/gpt2-medium")
tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token # padトークンの定義

# 学習用テキスト作成
def format_sst2(example):
    text=example["sentence"]
    label="positive" if example["label"]==1 else "negative"
    return f"Review: {text}\nSentiment: {label}"

train_texts = [format_sst2(row) for _, row in train_df.iterrows()]
dev_texts   = [format_sst2(row) for _, row in dev_df.iterrows()]

# データセット化
train_dataset=Dataset.from_dict({"text":train_texts})
dev_dataset=Dataset.from_dict({"text":dev_texts})

# tokenize
def tokenize_fn(examples):
    return tokenizer(examples["text"],truncation=True,padding="max_length",max_length=128)

train_dataset=train_dataset.map(tokenize_fn,batched=True)
train_dataset.est_format(type="torch",columns=["input_ids","attention_mask"])

dev_dataset = dev_dataset.map(tokenize_fn, batched=True)
dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# モデルの設定
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# 学習設定
training_args=TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# ファインチューニング
trainer.train()

prompt = "Review: This movie was great!\nSentiment:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=5)
print(tokenizer.decode(outputs[0]))
