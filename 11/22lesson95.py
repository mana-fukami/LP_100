"""
トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import math
import sacrebleu
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch.nn.functional as F
import heapq
import os
from functools import partial
from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# 1. DDP (分散学習) の設定
# =============================================================================
def setup(rank, world_size):
    """DDPのプロセスグループを初期化する"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """DDPのプロセスグループをクリーンアップする"""
    dist.destroy_process_group()

# =============================================================================
# 2. データセットとデータローダーの定義
# =============================================================================
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences
    def __len__(self):
        return len(self.src_sequences)
    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx], dtype=torch.long), torch.tensor(self.tgt_sequences[idx], dtype=torch.long)

def collate_fn(batch, sp_ja, sp_en):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=sp_ja.pad_id(), batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=sp_en.pad_id(), batch_first=True)
    return src_batch, tgt_batch

# =============================================================================
# 3. Transformerモデルの定義
# =============================================================================
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
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=3) # pad_id=3 を指定
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=3)
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

# =============================================================================
# 4. Worker関数
# =============================================================================
def worker(rank, world_size, ja_ids, en_ids, sp_ja, sp_en):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    train_dataset = TranslationDataset(ja_ids, en_ids)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    partial_collate_fn = partial(collate_fn, sp_ja=sp_ja, sp_en=sp_en)
    train_loader = DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler,
        collate_fn=partial_collate_fn, num_workers=2, pin_memory=True
    )

    model = TransformerNMT(sp_ja.get_piece_size(), sp_en.get_piece_size()).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=sp_en.pad_id())
    
    num_epochs = 10
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        optimizer.zero_grad() 

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
        for src_batch, tgt_batch in pbar:
            src_batch, tgt_batch = src_batch.to(rank), tgt_batch.to(rank)
            tgt_input, tgt_output = tgt_batch[:, :-1], tgt_batch[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), rank)
            src_key_padding_mask = (src_batch == sp_ja.pad_id())
            tgt_key_padding_mask = (tgt_input == sp_en.pad_id())

            with autocast():
                output = model(
                    src_batch, tgt_input,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                loss = criterion(output.view(-1, output.size(-1)), tgt_output.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # DDPで損失を集計
        loss_tensor = torch.tensor([total_loss / len(train_loader)]).to(rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            avg_loss = loss_tensor.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "subword_transformer_nmt.pt")
        print("Model saved to subword_transformer_nmt.pt")

    cleanup()

# =============================================================================
# 5. 評価と翻訳
# =============================================================================
def translate_beam_search(model, sentence, sp_ja, sp_en, device, beam_width=5, max_len=50):
    model.eval()
    tokens = sp_ja.encode_as_ids(sentence)
    src = torch.tensor([[sp_ja.bos_id()] + tokens + [sp_ja.eos_id()]], device=device)
    src_padding_mask = (src == sp_ja.pad_id())

    # [ (log_prob, sequence) ]
    beams = [(0.0, [sp_en.bos_id()])]
    
    for _ in range(max_len):
        new_beams = []
        for log_prob, seq in beams:
            if seq[-1] == sp_en.eos_id():
                new_beams.append((log_prob, seq))
                continue

            tgt_input = torch.tensor([seq], device=device)

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
            
            with torch.no_grad():
                out = model(src, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask)
            
            log_probs = F.log_softmax(out[0, -1, :], dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_width)

            for lp, idx in zip(topk_log_probs, topk_indices):
                new_beams.append((log_prob + lp.item(), seq + [idx.item()]))
        
        # スコアの高い順にビームをソートして、上位beam_width個だけを残す
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
        
        if all(b[1][-1] == sp_en.eos_id() for b in beams):
            break
            
    best_log_prob, best_seq = beams[0]
    return sp_en.decode_ids(best_seq[1:-1]) # <sos>と<eos>を除く


def evaluate_and_plot(model, device, sp_ja, sp_en):
    print("\n--- Starting evaluation ---")
    
    with open("./kftt-data-1.0/data/orig/kyoto-dev.ja", "r", encoding="utf-8") as f:
        dev_ja = [line.strip() for line in f]
    with open("./kftt-data-1.0/data/orig/kyoto-dev.en", "r", encoding="utf-8") as f:
        dev_en = [line.strip() for line in f]

    beam_widths = [1, 5, 10, 20, 50]
    bleu_scores = []
    for width in beam_widths:
        print(f"Evaluating with Beam Search (width={width})...")
        hypotheses = [translate_beam_search(model, s, sp_ja, sp_en, device, beam_width=width) for s in tqdm(dev_ja)]
        bleu = sacrebleu.corpus_bleu(hypotheses, [dev_en])
        print(f"Beam Width: {width}, BLEU: {bleu.score:.2f}")
        bleu_scores.append(bleu.score)

    plt.figure(figsize=(10, 6))
    plt.plot(beam_widths, bleu_scores, marker='o')
    plt.xlabel('Beam Width'); plt.ylabel('BLEU Score'); plt.title('Beam Width vs BLEU Score')
    plt.grid(True); plt.savefig('subword_beam_width_vs_bleu.png')
    plt.close()
    print("Plot saved to subword_beam_width_vs_bleu.png")

# =============================================================================
# 6. メイン実行ブロック
# =============================================================================
def main():
    ja_model_file = "ja_spm.model"
    en_model_file = "en_spm.model"
    
    if not (os.path.exists(ja_model_file) and os.path.exists(en_model_file)):
        print("Training SentencePiece models...")
        spm.SentencePieceTrainer.train(
            input="./kftt-data-1.0/data/tok/kyoto-train.ja", model_prefix="ja_spm",
            vocab_size=16000, character_coverage=1.0, model_type="bpe", pad_id=3
        )
        spm.SentencePieceTrainer.train(
            input="./kftt-data-1.0/data/tok/kyoto-train.en", model_prefix="en_spm",
            vocab_size=16000, character_coverage=1.0, model_type="bpe", pad_id=3
        )

    sp_ja = spm.SentencePieceProcessor(model_file=ja_model_file)
    sp_en = spm.SentencePieceProcessor(model_file=en_model_file)

    with open("./kftt-data-1.0/data/tok/kyoto-train.ja", "r", encoding="utf-8") as f:
        train_ja_lines = f.readlines()
    with open("./kftt-data-1.0/data/tok/kyoto-train.en", "r", encoding="utf-8") as f:
        train_en_lines = f.readlines()

    ja_ids = [[sp_ja.bos_id()] + sp_ja.encode_as_ids(s) + [sp_ja.eos_id()] for s in train_ja_lines]
    en_ids = [[sp_en.bos_id()] + sp_en.encode_as_ids(s) + [sp_en.eos_id()] for s in train_en_lines]

    world_size = torch.cuda.device_count()
    if world_size > 1:
        args = (world_size, ja_ids, en_ids, sp_ja, sp_en)
        mp.spawn(worker, args=args, nprocs=world_size, join=True)
    else:
        print("This script is designed for DDP with multiple GPUs.")
        # シングルGPU用の簡易的な学習・評価パスを追加しても良い
        print("Please run with at least 2 GPUs.")
        return # シングルGPUでは実行しない

    # --- 評価はメインプロセスのみが実行 ---
    print("\nDDP training finished. Starting evaluation on the main process.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_model = TransformerNMT(sp_ja.get_piece_size(), sp_en.get_piece_size()).to(device)
    eval_model.load_state_dict(torch.load("subword_transformer_nmt.pt", map_location=device))
    
    evaluate_and_plot(eval_model, device, sp_ja, sp_en)

if __name__ == '__main__':
    main()
