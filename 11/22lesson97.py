"""
ニューラルネットワークのモデルや,そのハイパーパラメータを変更しつつ,開発データにおけるBLEUスコアが
最大となるモデルとハイパーパラメータを求めよ．
チューニングを行うハイパーパラメータは項目は以下とすること
    バッチサイズ
    学習率(1e-3~1e-6)
    Optimizer (Adam, AdamW, RAdamなどAdam系をいくつか)
最終的にBLUEスコアは10以上になることを確認すること
"""
<<<<<<< HEAD
=======
import wandb
>>>>>>> bf4cb87d849b5ce0466928ebd9c9f3c86a5d0ead
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import math
import os
<<<<<<< HEAD
import sacrebleu
=======
from nltk.translate.bleu_score import sentence_bleu
>>>>>>> bf4cb87d849b5ce0466928ebd9c9f3c86a5d0ead
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# GPUに移動する
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# データの準備をする
# -トークン化されたファイルを開く
en_file=open("./kftt-data-1.0/data/tok/kyoto-train.en","r",encoding="utf-8")
en_lines=en_file.readlines()
ja_file=open("./kftt-data-1.0/data/tok/kyoto-train.ja","r",encoding="utf-8")
ja_lines=ja_file.readlines()

# -トークン列のリストを作る
en_tokenized=[]
for line in en_lines:
    en_tokenized.append(line.strip().split())
ja_tokenized=[]
for line in ja_lines:
    ja_tokenized.append(line.strip().split())

# -語彙の作成
en_counter=Counter()
ja_counter=Counter()
for tokens in en_tokenized:
    en_counter.update(tokens)
for tokens in ja_tokenized:
    ja_counter.update(tokens)

# -頻度の高い順に並べる
max_vocab_size = 50000
en_vocab_list = ['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in en_counter.most_common(max_vocab_size)]
ja_vocab_list=['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in ja_counter.most_common(max_vocab_size)]
#print(f"vocab_size: {len(en_vocab_list)}")

# -IDの辞書
en_token2id = {token: idx for idx, token in enumerate(en_vocab_list)}
ja_token2id = {token: idx for idx, token in enumerate(ja_vocab_list)}

# -逆引き
en_id2token = {idx: token for token, idx in en_token2id.items()}
ja_id2token = {idx: token for token, idx in ja_token2id.items()}

# -数値列に変換
en_ids=[]
ja_ids=[]
for ja in ja_tokenized:
    ids = [ja_token2id.get(token, ja_token2id['<unk>']) for token in ja]
    ja_ids.append( [ja_token2id['<sos>']] + ids + [ja_token2id['<eos>']] )
for en in en_tokenized:
    ids = [en_token2id.get(token, en_token2id['<unk>']) for token in en]
    en_ids.append( [en_token2id['<sos>']] + ids + [en_token2id['<eos>']] )

# -データセット化
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences  # ja_ids
        self.tgt_sequences = tgt_sequences  # en_ids

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx], dtype=torch.long),torch.tensor(self.tgt_sequences[idx], dtype=torch.long)

def Collate(batch):
    src_batch, tgt_batch = zip(*batch)  # バッチの中のサンプルを分ける
    src_batch = pad_sequence(src_batch, padding_value=ja_token2id['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=en_token2id['<pad>'], batch_first=True)
    return src_batch, tgt_batch

# 文の順序を保持するための位置エンコーディング
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        # d_model:埋め込み層の次元数, nhead:マルチヘッドアテンションのヘッド数
        # num_layers:エンコーダ・デコーダの層の数, dim__ff:feed-forwardの中間層の次元数
        super().__init__()
        # 日本語と単語のIDをベクトルに変化する
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # Transformerは順序情報を持たないので、位置エンコーディングを行う
        self.positional_encoding = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout
        )
        # d_model次元の特徴ベクトルをターゲット語彙数に変換する
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embedding + positional
        # ID列をベクトルにし、位置エンコーディングを加える
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        
        # transformer expects [seq_len, batch_size, d_model]
        # Transformerに合わせてデータを整形する
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # output: [seq_len, batch_size, d_model]
        # Transformerの出力を元のデータに合わせる
        output = output.transpose(0, 1)  # back to [batch_size, seq_len, d_model]
        # 最後に線形層を通して語彙数に変換する
        return self.output_layer(output)

# マスクの生成
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def calculate_bleu_score(model, data_loader, ja_id2token, en_id2token, device):
    model.eval()
    total_bleu = 0
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            tgt_input = tgt_batch[:, :-1]  # decoder入力
            tgt_output = tgt_batch[:, 1:]  # decoder出力の正解

            # 推論
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_pad_id = ja_token2id['<pad>']
            tgt_pad_id = en_token2id['<pad>']

            output = model(
                src_batch,
                tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=(src_batch == src_pad_id),
                tgt_padding_mask=(tgt_input == tgt_pad_id),
                memory_key_padding_mask=(src_batch == src_pad_id)
            )

            # 出力をデコード
            output = torch.argmax(output, dim=-1)  # [batch_size, tgt_len]
            for pred, target in zip(output, tgt_output):
                pred_tokens = [en_id2token[idx.item()] for idx in pred if idx.item() not in [en_token2id['<pad>'], en_token2id['<sos>'], en_token2id['<eos>']]]
                target_tokens = [en_id2token[idx.item()] for idx in target if idx.item() not in [en_token2id['<pad>'], en_token2id['<sos>'], en_token2id['<eos>']]]
<<<<<<< HEAD
                total_bleu += sacrebleu.corpus_bleu([target_tokens], pred_tokens)
=======
                total_bleu += sentence_bleu([target_tokens], pred_tokens)
>>>>>>> bf4cb87d849b5ce0466928ebd9c9f3c86a5d0ead

    return total_bleu / len(data_loader)

# 学習ループ
src_vocab_size = len(ja_token2id)
tgt_vocab_size = len(en_token2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNMT(
    src_vocab_size,
    tgt_vocab_size,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_ff=2048
).to(device)
#model = nn.DataParallel(model)  # DataParallelでラップ
pad_id = en_token2id['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

<<<<<<< HEAD
epochs=10
def train_roop(train_loader,optimizer):
    model.train()
    for epoch in range(epochs):
=======
# Wndbのログイン
key=open("AllKeys/wandb.txt").readline()
wandb.login(key=key)

# Wndbの初期化
wandb.init(project="22lesson97")

# ハイパーパラメータの設定
wandb.config.epochs=20

def train_roop(train_loader,optimizer):
    model.train()
    for epoch in range(2,wandb.config.epochs):
>>>>>>> bf4cb87d849b5ce0466928ebd9c9f3c86a5d0ead
        model.train()
        total_loss = 0
        
        # tqdmで進捗バーを作成
        for src_batch, tgt_batch in tqdm(train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            tgt_input = tgt_batch[:, :-1]  # decoder入力
            tgt_output = tgt_batch[:, 1:]  # decoder出力の正解

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            
            src_pad_id = ja_token2id['<pad>']
            tgt_pad_id = en_token2id['<pad>']
            
            optimizer.zero_grad()
            
            output = model(
                src_batch,
                tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=(src_batch == src_pad_id),
                tgt_padding_mask=(tgt_input == tgt_pad_id),
                memory_key_padding_mask=(src_batch == src_pad_id)
            )
            
            # 出力 shape [batch_size, tgt_len, vocab_size] → [batch_size * tgt_len, vocab_size]
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

batch_size=[32,64]
learning_rates = [1e-3, 1e-4, 1e-5]
optimizers = ['Adam', 'AdamW', 'RAdam']

def get_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_name == 'RAdam':
        return torch.optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

result_df = pd.DataFrame(columns=['batch_size', 'learning_rate', 'optimizer', 'bleu_score'])
for batch in batch_size:
    train_dataset=TranslationDataset(ja_ids,en_ids)
    train_loader=DataLoader(train_dataset,batch_size=batch,shuffle=True,collate_fn=Collate,num_workers=4)
    for lr in learning_rates:
        for opt in optimizers:
            optimizer = get_optimizer(opt,lr)
            train_roop(train_loader=train_loader,optimizer=optimizer)
            blue_score= calculate_bleu_score(model, train_loader, ja_id2token, en_id2token, device)
            result_df["batch_size"]=batch
            result_df["learning_rate"]=lr
            result_df["optimizer"]=opt
            result_df["bleu_score"]=blue_score
            # 学習率を文字列にして見やすくする
            result_df["lr_str"] = result_df["lr"].apply(lambda x: f"{x:.0e}")

# 結果のヒートマップ
g = sns.FacetGrid(result_df, col="optimizer", height=4)
g.map_dataframe(
    lambda data, color: sns.heatmap(
        data.pivot("batch_size", "lr_str", "bleu"),
        annot=True, fmt=".2f", cmap="viridis", cbar=False
    )
)
plt.suptitle("BLEUスコア比較（各オプティマイザ）", y=1.05)
plt.savefig('hyper_tuning.png', dpi=300, bbox_inches='tight')
plt.close()