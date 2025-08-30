import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import sacrebleu

# -データセット化
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx], dtype=torch.long), torch.tensor(self.tgt_sequences[idx], dtype=torch.long)

def Collate(batch, ja_token2id, en_token2id):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=ja_token2id['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=en_token2id['<pad>'], batch_first=True)
    return src_batch, tgt_batch

# モデル定義 (元のコードと同じ)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer(
            src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        output = output.transpose(0, 1)
        return self.output_layer(output)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# 勾配蓄積+混合精度学習を適用
def train_loop(rank, model, train_loader, optimizer, criterion, epochs, ja_token2id, en_token2id):
    """1試行あたりの学習ループ (勾配蓄積を実装)"""
    if rank == 0: print("Train loop started!")
    model.train()
    accumulation_steps = 4  # 4ステップで1回パラメータを更新 (実質的なバッチサイズが4倍に)
    scaler = GradScaler()

    for epoch in range(epochs):
        # DistributedSamplerを使う場合、各エポックでset_epochを呼び出す必要がある
        train_loader.sampler.set_epoch(epoch)
        # tqdmをマスタープロセス(rank 0)でのみ表示する
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, (src_batch, tgt_batch) in enumerate(pbar):
            src_batch = src_batch.to(rank)
            tgt_batch = tgt_batch.to(rank)
            
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(rank)
            
            src_pad_id = ja_token2id['<pad>']
            tgt_pad_id = en_token2id['<pad>']
            
            with autocast():
                output = model(
                    src_batch, tgt_input,
                    tgt_mask=tgt_mask,
                    src_padding_mask=(src_batch == src_pad_id),
                    tgt_padding_mask=(tgt_input == tgt_pad_id),
                    memory_key_padding_mask=(src_batch == src_pad_id)
                )
                
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(output, tgt_output)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if rank == 0:
                pbar.set_postfix(loss=loss.item() * accumulation_steps)
        
        # エポックの最後に更新されなかった勾配を更新
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

# BLEUスコア計算 (変更なし)
def calculate_bleu_score(model, data_loader, device, ja_token2id, en_token2id, en_id2token):
    # DDPでラップされている場合、元のモデルにアクセスするために .module を使用
    model_to_eval = model.module if isinstance(model, DDP) else model
    model_to_eval.eval()
    
    refs = []
    hyps = []

    with torch.no_grad():
        for src_batch, tgt_batch in data_loader:
            src_batch = src_batch.to(device)
            
            # 貪欲法(Greedy Search)によるデコード
            # <sos>トークンから開始
            memory = model_to_eval.transformer.encoder(
                model_to_eval.positional_encoding(model_to_eval.src_embedding(src_batch)).transpose(0,1),
                src_key_padding_mask=(src_batch == ja_token2id['<pad>'])
            )

            batch_size = src_batch.size(0)
            ys = torch.ones(batch_size, 1).fill_(en_token2id['<sos>']).long().to(device)
            
            for _ in range(100): # 最大生成長
                tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)
                out = model_to_eval.transformer.decoder(
                    model_to_eval.positional_encoding(model_to_eval.tgt_embedding(ys)).transpose(0,1), 
                    memory, 
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                out = out.transpose(0,1)
                prob = model_to_eval.output_layer(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if torch.all(next_word.squeeze() == en_token2id['<eos>']):
                    break

            # IDをトークンに変換
            for i in range(ys.size(0)):
                pred_ids = ys[i].cpu().numpy()
                pred_tokens = [en_id2token[idx] for idx in pred_ids if idx not in [en_token2id['<pad>'], en_token2id['<sos>'], en_token2id['<eos>']]]
                hyps.append(" ".join(pred_tokens))

            for i in range(tgt_batch.size(0)):
                target_ids = tgt_batch[i].cpu().numpy()
                target_tokens = [en_id2token[idx] for idx in target_ids if idx not in [en_token2id['<pad>'], en_token2id['<sos>'], en_token2id['<eos>']]]
                refs.append([" ".join(target_tokens)])
    
    # sacrebleuは参照訳をリストのリストとして受け取る
    bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='none')
    return bleu.score

def main():
    # データの準備
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
    max_vocab_size = 25000
    en_vocab_list = ['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in en_counter.most_common(max_vocab_size)]
    ja_vocab_list=['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in ja_counter.most_common(max_vocab_size)]

    # -IDの辞書
    en_token2id = {token: idx for idx, token in enumerate(en_vocab_list)}
    ja_token2id = {token: idx for idx, token in enumerate(ja_vocab_list)}
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

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    args = parser.parse_args()

    # DDPの初期化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # モデルやパラメータの設定
    # データセットの準備
    train_dataset = TranslationDataset(ja_ids, en_ids)
    
    # DDPではDistributedSamplerを使用してデータを各プロセスに分割
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    
    # DataLoaderの作成
    # DDPの場合、shuffle=Falseにする (Samplerがシャッフルするため)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=lambda batch: Collate(batch, ja_token2id, en_token2id),
        sampler=train_sampler, pin_memory=True, num_workers=0
    )

    # モデルの初期化
    src_vocab_size = len(ja_token2id)
    tgt_vocab_size = len(en_token2id)
    model = TransformerNMT(
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_ff=2048
    ).to(device)
    # モデルをDDPでラップ
    model = DDP(model, device_ids=[rank])

    # Optimizerの選択
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pad_id = en_token2id['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # 学習実行
    epochs = 5  # 1回の試行でのエポック数 (適宜調整)
    train_loop(rank, model, train_loader, optimizer, criterion, epochs, ja_token2id, en_token2id)

    # マスタープロセス(rank 0)のみで評価と結果報告を行う
    if rank == 0:
        # 評価用のデータローダ (Samplerなし)
        # サンプル数を減らして評価を高速化することも可能
        val_dataset = TranslationDataset(ja_ids[:500], en_ids[:500]) # 500文で評価
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            collate_fn=lambda batch: Collate(batch, ja_token2id, en_token2id))
        
        bleu_score = calculate_bleu_score(model, val_loader, rank, ja_token2id, en_token2id, en_id2token)
        print(f"Trial finished. BLEU Score: {bleu_score}")
        # 結果を出力する（メインプロセスに引き渡す）
        print(f"{bleu_score}")

    dist.destroy_process_group()

if __name__=="__main__":
    main()    