"""
Japanese-English Subtitle Corpus (JESC)やJParaCrawlなどの
翻訳データを活用し，KFTTのテストデータの性能向上を試みよ．
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import math
import os
import MeCab
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# GPUに移動する
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# データの準備をする
# -ファイルを開く
en_ja_file=open("./split/train","r",encoding="utf-8")
en_ja_lines=en_ja_file.readlines()
en_lines=[]
ja_lines=[]
for line in en_ja_lines:
    en_ja=line.strip().split("\t")
    en_lines.append(en_ja[0])
    ja_lines.append(en_ja[1])

# -トークン列のリストを作る
tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
ja_tokenized=[]
for line in ja_lines:
    node=tagger.parseToNode(line.strip())
    tokens = []
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    ja_tokenized.append(tokens)
en_tokenized=[]
for line in en_lines:
    tokens=line.strip().split(" ")
    en_tokenized.append(tokens)

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

train_dataset=TranslationDataset(ja_ids,en_ids)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=Collate,num_workers=4)

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
model.load_state_dict(torch.load("transformer_nmt.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.98), eps=1e-9)
pad_id = en_token2id['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

num_epochs = 20
for epoch in range(num_epochs):
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
    
    print(f"epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")
    torch.cuda.empty_cache()

# モデルと語彙の保存(次で使えるように)
torch.save(model.state_dict(), "add_train_transformer_nmt.pt")