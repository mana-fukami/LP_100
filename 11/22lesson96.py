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
import pickle # モデル保存用にインポート

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

# --- データ準備 ---
def load_data(en_path, ja_path):
    with open(en_path,"r",encoding="utf-8") as f:
        en_lines=f.readlines()
    with open(ja_path,"r",encoding="utf-8") as f:
        ja_lines=f.readlines()
    en_tokenized = [line.strip().split() for line in en_lines]
    ja_tokenized = [line.strip().split() for line in ja_lines]
    return en_tokenized, ja_tokenized

train_en_tokenized, train_ja_tokenized = load_data("./kftt-data-1.0/data/tok/kyoto-train.en", "./kftt-data-1.0/data/tok/kyoto-train.ja")
dev_en_tokenized, dev_ja_tokenized = load_data("./kftt-data-1.0/data/tok/kyoto-dev.en", "./kftt-data-1.0/data/tok/kyoto-dev.ja")

en_counter = Counter(token for tokens in train_en_tokenized for token in tokens)
ja_counter = Counter(token for tokens in train_ja_tokenized for token in tokens)

max_vocab_size = 50000
en_vocab_list = ['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in en_counter.most_common(max_vocab_size)]
ja_vocab_list=['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in ja_counter.most_common(max_vocab_size)]

en_token2id = {token: idx for idx, token in enumerate(en_vocab_list)}
ja_token2id = {token: idx for idx, token in enumerate(ja_vocab_list)}
en_id2token = {idx: token for token, idx in en_token2id.items()}
ja_id2token = {idx: token for token, idx in ja_token2id.items()}

def tokens_to_ids(tokenized_sentences, token2id):
    sos_id, eos_id, unk_id = token2id['<sos>'], token2id['<eos>'], token2id['<unk>']
    return [[sos_id] + [token2id.get(token, unk_id) for token in tokens] + [eos_id] for tokens in tokenized_sentences]

train_ja_ids, train_en_ids = tokens_to_ids(train_ja_tokenized, ja_token2id), tokens_to_ids(train_en_tokenized, en_token2id)
dev_ja_ids, dev_en_ids = tokens_to_ids(dev_ja_tokenized, ja_token2id), tokens_to_ids(dev_en_tokenized, en_token2id)

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences, self.tgt_sequences = src_sequences, tgt_sequences
    def __len__(self):
        return len(self.src_sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx], dtype=torch.long), torch.tensor(self.tgt_sequences[idx], dtype=torch.long)

def Collate(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, padding_value=ja_token2id['<pad>'], batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=en_token2id['<pad>'], batch_first=True)
    return src_padded, tgt_padded

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(output)

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def validate(model, data_loader, criterion, rank):
    model.eval()
    total_loss = 0
    local_preds, local_targets_for_bleu = [], []
    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch, tgt_batch = src_batch.to(rank), tgt_batch.to(rank)
            tgt_input, tgt_for_loss, tgt_for_bleu = tgt_batch[:, :-1], tgt_batch[:, 1:].reshape(-1), tgt_batch[:, 1:]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), rank)
            src_key_padding_mask = (src_batch == ja_token2id['<pad>'])
            tgt_key_padding_mask = (tgt_input == en_token2id['<pad>'])

            output = model(src_batch, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_for_loss)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=-1)
            for p, t in zip(preds, tgt_for_bleu):
                pred_tokens = [en_id2token.get(idx.item(), '<unk>') for idx in p if idx.item() != en_token2id['<pad>']]
                target_tokens = [en_id2token.get(idx.item(), '<unk>') for idx in t if idx.item() != en_token2id['<pad>']]
                try: pred_tokens = pred_tokens[:pred_tokens.index('<eos>')]
                except ValueError: pass
                local_preds.append(" ".join(pred_tokens))
                local_targets_for_bleu.append(" ".join(target_tokens))

    # 全プロセスの結果を集約
    gathered_preds, gathered_targets = [None] * dist.get_world_size(), [None] * dist.get_world_size()
    dist.all_gather_object(gathered_preds, local_preds)
    dist.all_gather_object(gathered_targets, local_targets_for_bleu)
    
    loss_tensor = torch.tensor([total_loss / len(data_loader)]).to(rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    avg_val_loss = loss_tensor.item()
    
    bleu_score = 0.0
    if rank == 0:
        all_preds = [p for sublist in gathered_preds for p in sublist]
        all_targets = [t for sublist in gathered_targets for t in sublist]
        bleu = sacrebleu.corpus_bleu(all_preds, [all_targets])
        bleu_score = bleu.score

    # 全プロセスが同じ値を返すようにする
    # rank 0 で計算したBLEUスコアを全プロセスにブロードキャスト（共有）する
    bleu_tensor = torch.tensor([bleu_score]).to(rank)
    dist.broadcast(bleu_tensor, src=0)
    
    return avg_val_loss, bleu_tensor.item()

def main_worker(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    train_dataset, dev_dataset = TranslationDataset(train_ja_ids, train_en_ids), TranslationDataset(dev_ja_ids, dev_en_ids)
    train_sampler, dev_sampler = DistributedSampler(train_dataset, shuffle=True), DistributedSampler(dev_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=Collate, sampler=train_sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=32, collate_fn=Collate, sampler=dev_sampler)

    model = TransformerNMT(len(ja_token2id), len(en_token2id)).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=en_token2id['<pad>'])

    if rank == 0:
        wandb.login()
        wandb.init(project="NMT-KFTT-DDP-Fixed", config={"epochs": 20, "batch_size": 32 * world_size})

    for epoch in range(20):
        model.train()
        train_sampler.set_epoch(epoch)
        total_train_loss = 0
        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        
        for src_batch, tgt_batch in pbar:
            src_batch, tgt_batch = src_batch.to(rank), tgt_batch.to(rank)
            tgt_input, tgt_output = tgt_batch[:, :-1], tgt_batch[:, 1:].reshape(-1)
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), rank)
            src_key_padding_mask = (src_batch == ja_token2id['<pad>'])
            tgt_key_padding_mask = (tgt_input == en_token2id['<pad>'])

            optimizer.zero_grad()
            output = model(src_batch, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # DDPでの損失集計
        loss_tensor = torch.tensor([total_train_loss]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        if rank == 0:
            avg_train_loss = loss_tensor.item() / (len(train_loader) * world_size)
        
        # 評価ステップ
        val_loss, val_bleu = validate(model, dev_loader, criterion, rank)

        if rank == 0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:.2f}")
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": val_loss, "val_bleu": val_bleu})
        
        dist.barrier() # オプション: 全プロセスがエポックの終わりで同期するのを保証
        
    if rank == 0:
        print("Saving model and vocabularies...")
        torch.save(model.module.state_dict(), "transformer_nmt_wandb.pt")
        with open("ja_token2id_wandb.pkl", "wb") as f:
            pickle.dump(ja_token2id, f)
        with open("en_token2id_wandb.pkl", "wb") as f:
            pickle.dump(en_token2id, f)
        with open("en_id2token_wandb.pkl", "wb") as f:
            pickle.dump(en_id2token, f)
        wandb.finish()
        
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1: # DDPは複数GPUでの実行を前提
        print(f"Found {world_size} GPUs. Spawning DDP processes.")
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("DDP requires at least 2 GPUs. This script will not run on a single GPU.")
