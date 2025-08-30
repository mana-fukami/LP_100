"""
Tensorboardなどのツールを用い,ニューラル機械翻訳モデルが学習されていく過程を可視化せよ.
可視化する項目としては,
    - 学習データにおける損失関数の値とBLEUスコア
    - 開発データにおける損失関数の値とBLEUスコア
などを採用せよ.
可視化はTensorboard or [WandB https://www.wandb.jp/]を使うこと (WandBのほうが簡単なのでおすすめ)
"""
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import math
import os
import sacrebleu

# DDPのライブラリ
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# DDPの初期化とクリーンアップ
def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank,world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

def calculate_bleu_score(model, data_loader, rank):
    model.eval()
    local_preds = []
    local_targets = []

    # DDPの各プロセスで推論を行う
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(rank)
            tgt_batch = tgt_batch.to(rank)

            tgt_input = tgt_batch[:, :-1]  # decoder入力
            tgt_output = tgt_batch[:, 1:]  # decoder出力の正解

            # 推論
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(rank)
            src_pad_id = ja_token2id['<pad>']
            tgt_pad_id = en_token2id['<pad>']

            output = model.module(
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
                # <pad>トークンを除去
                pred_tokens = [en_id2token[idx.item()] for idx in pred if idx.item() != en_token2id['<pad>']]
                target_tokens = [en_id2token[idx.item()] for idx in target if idx.item() != en_token2id['<pad>']]
                # <sos> <eos> を除去
                pred_tokens = [tok for tok in pred_tokens if tok not in ['<sos>', '<eos>']]
                target_tokens = [tok for tok in target_tokens if tok not in ['<sos>', '<eos>']]
                local_preds.append(" ".join(pred_tokens))
                local_targets.append(" ".join(target_tokens))
    # 各プロセスの推論結果をリストにまとめる
    gathered_preds = [None] * dist.get_world_size()
    gathered_targets = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_preds, local_preds)
    dist.all_gather_object(gathered_targets, local_targets)
    # rank 0 のプロセスのみでBLEUスコアを計算
    if rank == 0:
        # リストをフラット化
        all_preds = [pred for sublist in gathered_preds for pred in sublist]
        all_targets = [target for sublist in gathered_targets for target in sublist]
        
        # sacrebleuは参照訳をリストのリストとして受け取る [[ref1], [ref2], ...]
        bleu = sacrebleu.corpus_bleu(all_preds, [all_targets])
        return bleu.
    
    return 0.0 # 他のプロセスはダミー値を返す

def main_worker(rank, world_size):
    # 学習ループ
    print(f"Running DDP on rank {rank}.")
    setup(rank,world_size)
    # データセットとサンプラーの準備
    train_dataset=TranslationDataset(ja_ids, en_ids)
    train_sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader=DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=Collate, num_workers=0, sampler=train_sampler)
    # モデルの準備
    src_vocab_size = len(ja_token2id)
    tgt_vocab_size = len(en_token2id)
    model = TransformerNMT(
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_ff=2048
    ).to(rank)
    # DDPでラップ
    model = DDP(model,device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.98), eps=1e-9)
    pad_id = en_token2id['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Wandbの記録(rank=0のみ)
    if rank==0:
        try:
            # Wndbのログイン
            key=open("AllKeys/wandb").readline()
            wandb.login(key=key)
            except Exception as e:
                print(f"Could not log in to WandB: {e}")
        
        # 初期化
        wandb.init(project="22lesson96")

        # ハイパーパラメータの設定
        wandb.config.epochs=20
        wandb.config.batch_size=32
        wandb.config.world_size=world_size
        
    # 学習ループ
    epochs=20
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        
        # rank=0のみでtqdm進捗バーを作成
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for src_batch, tgt_batch in pbar:
            src_batch = src_batch.to(rank)
            tgt_batch = tgt_batch.to(rank)
            
            tgt_input = tgt_batch[:, :-1]  # decoder入力
            tgt_output = tgt_batch[:, 1:]  # decoder出力の正解

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(rank)
            
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

        # 全プロセスでロスを同期・平均化
        avg_loss = total_loss / len(train_loader)
        loss_tensor = torch.tensor([avg_loss]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss_all = loss_tensor.item()

        # BLUEスコアの計算
        bleu_score = calculate_bleu_score(
            model,
            train_loader,
            rank
        )
        # rank==0のみでWandbへの記録
        if rank==0:
            wandb.log(
                {
                    "bleu_score":bleu_score,
                    "loss":total_loss/len(train_loader),
                }
            )
            print(f"epoch {epoch+1} loss: {total_loss / len(train_loader):.4f}")
        torch.cuda.empty_cache()
    if rank==0:
        wandb.finish()
    cleanup()

if __name__ == '__main__':
    world_size = torch.cude.device_count()
    if world_size > 0:
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        # spawn を使って DDP プロセスを起動
        mp.spawn(main_worker,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        print("No GPUs found. DDP requires GPUs.")
"""
# モデルと語彙の保存(次で使えるように)
torch.save(model.state_dict(), "transformer_nmt.pt")
import pickle
with open("ja_token2id.pkl", "wb") as f:
    pickle.dump(ja_token2id, f)
with open("en_token2id.pkl", "wb") as f:
    pickle.dump(en_token2id, f)
with open("en_id2token.pkl", "wb") as f:
    pickle.dump(en_id2token, f)
"""