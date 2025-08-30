"""
トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
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
    """バッチ内のシーケンスをパディングしてテンソルにまとめる"""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=sp_ja.pad_id(), batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=sp_en.pad_id(), batch_first=True)
    return src_batch, tgt_batch

# =============================================================================
# 3. Transformerモデルの定義
# =============================================================================
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
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True # batch_first=Trueに設定
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        
        # Transformerはデフォルトでbatch_first=Falseなのでtransposeは不要
        # ただし、nn.Transformerのコンストラクタでbatch_first=Trueにすると、このtransposeは不要になる
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(output)

def generate_square_subsequent_mask(sz, device):
    """未来のトークンを見ないようにするためのマスクを生成"""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# =============================================================================
# 4. Worker関数
# =============================================================================
def worker(rank, world_size, ja_ids, en_ids, sp_ja, sp_en):
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)

    # データセットとサンプラー、データローダーの準備
    train_dataset = TranslationDataset(ja_ids, en_ids)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # collate_fnにspmプロセッサを渡すためにlambdaを使用
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, sp_ja, sp_en),
        num_workers=2, pin_memory=True
    )

    # モデルとDDPの準備
    src_vocab_size = sp_ja.get_piece_size()
    tgt_vocab_size = sp_en.get_piece_size()

    model = TransformerNMT(src_vocab_size, tgt_vocab_size).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=sp_en.pad_id())

    num_epochs = 10
    for epoch in range(num_epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch) # shuffleが正しく機能するために必要
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
        for src_batch, tgt_batch in pbar:
            # 修正点: テンソルはrankに送る
            src_batch = src_batch.to(rank)
            tgt_batch = tgt_batch.to(rank)

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), rank)
            src_key_padding_mask = (src_batch == sp_ja.pad_id())
            tgt_key_padding_mask = (tgt_input == sp_en.pad_id())

            optimizer.zero_grad()
            output = ddp_model(
                src_batch, tgt_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            if rank == 0:
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "subword_transformer_nmt_ddp.pt")
        print("Model saved to subword_transformer_nmt_ddp.pt")

    cleanup()

# =============================================================================
# 5. 評価と翻訳
# =============================================================================
def translate_greedy(model, sentence, sp_ja, sp_en, device, max_len=50):
    """Greedy Searchによる翻訳"""
    model.eval()
    tokens = sp_ja.encode_as_ids(sentence)
    src = torch.tensor([[sp_ja.bos_id()] + tokens + [sp_ja.eos_id()]], device=device)
    src_padding_mask = (src == sp_ja.pad_id())
    
    generated_ids = [sp_en.bos_id()]
    for _ in range(max_len):
        tgt_input = torch.tensor([generated_ids], device=device)
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
        
        with torch.no_grad():
            out = model(src, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=src_padding_mask)
        
        next_token_id = out[0, -1].argmax(-1).item()
        generated_ids.append(next_token_id)
        if next_token_id == sp_en.eos_id():
            break
            
    return sp_en.decode_ids(generated_ids[1:-1]) # <sos>と<eos>を除く

class BeamNode:
    def __init__(self, seq, logprob, length):
        self.seq = seq
        self.logprob = logprob
        self.length = length

    def __lt__(self, other):
        # 長さで正規化したスコアで比較
        return self.logprob / self.length < other.logprob / other.length

def translate_beam_search(model, sentence, sp_ja, sp_en, device, beam_width=5, max_len=50):
    """Beam Searchによる翻訳"""
    model.eval()
    tokens = sp_ja.encode_as_ids(sentence)
    src = torch.tensor([[sp_ja.bos_id()] + tokens + [sp_ja.eos_id()]], device=device)
    src_padding_mask = (src == sp_ja.pad_id())

    start_node = BeamNode([sp_en.bos_id()], 0.0, 1)
    beams = [start_node]
    completed_beams = []

    for _ in range(max_len):
        new_beams = []
        for node in beams:
            # 既に<eos>で終わっている場合は、次のステップに進めない
            if node.seq[-1] == sp_en.eos_id():
                completed_beams.append(node)
                continue

            tgt_input = torch.tensor([node.seq], device=device)
            with torch.no_grad():
                out = model(src, tgt_input, src_key_padding_mask=src_padding_mask)
            
            log_probs = F.log_softmax(out[:, -1, :], dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_width)

            for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                new_seq = node.seq + [idx.item()]
                new_node = BeamNode(new_seq, node.logprob + log_prob.item(), len(new_seq))
                new_beams.append(new_node)
        
        # 完了したビームと新しいビームを合わせてスコアの高いものを選ぶ
        all_candidates = completed_beams + new_beams
        beams = heapq.nlargest(beam_width, all_candidates, key=lambda x: x.logprob / x.length)

        # 全てのビームが<eos>で終わったら探索終了
        if all(b.seq[-1] == sp_en.eos_id() for b in beams):
            break
            
    best_beam = max(beams, key=lambda x: x.logprob / x.length)
    return sp_en.decode_ids(best_beam.seq[1:-1])


def evaluate_and_plot(model, device, sp_ja, sp_en):
    """モデルを評価し、ビーム幅とBLEUスコアの関係をプロットする"""
    print("\n--- Starting evaluation ---")
    
    # 開発データ読み込み
    with open("./kftt-data-1.0/data/orig/kyoto-dev.ja", "r", encoding="utf-8") as f:
        dev_ja = [line.strip() for line in f]
    with open("./kftt-data-1.0/data/orig/kyoto-dev.en", "r", encoding="utf-8") as f:
        dev_en = [line.strip() for line in f]

    # Greedy Searchの評価
    print("Evaluating with Greedy Search...")
    hypotheses_greedy = [translate_greedy(model, s, sp_ja, sp_en, device) for s in tqdm(dev_ja)]
    bleu_greedy = sacrebleu.corpus_bleu(hypotheses_greedy, [dev_en])
    print(f"Greedy Search BLEU: {bleu_greedy.score:.2f}")

    # Beam Searchの評価
    beam_widths = [1, 5, 10, 20, 50]
    bleu_scores = []
    for width in beam_widths:
        print(f"Evaluating with Beam Search (width={width})...")
        hypotheses_beam = [translate_beam_search(model, s, sp_ja, sp_en, device, beam_width=width) for s in tqdm(dev_ja)]
        bleu = sacrebleu.corpus_bleu(hypotheses_beam, [dev_en])
        print(f"Beam Width: {width}, BLEU: {bleu.score:.2f}")
        bleu_scores.append(bleu.score)

    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(beam_widths, bleu_scores, marker='o')
    plt.xlabel('Beam Width')
    plt.ylabel('BLEU Score')
    plt.title('Beam Width vs BLEU Score on Dev Set')
    plt.grid(True)
    plt.savefig('subword_beam_width_vs_bleu.png')
    plt.close()
    print("Plot saved to subword_beam_width_vs_bleu.png")


# =============================================================================
# 6. メイン実行ブロック
# =============================================================================
def main():
    # --- データ準備とSentencePieceモデルの学習 ---
    # SentencePieceモデルの学習は初回のみ実行すれば良い
    ja_model_file = "ja_spm.model"
    en_model_file = "en_spm.model"
    
    if not os.path.exists(ja_model_file):
        print("Training Japanese SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input="./kftt-data-1.0/data/tok/kyoto-train.ja",
            model_prefix="ja_spm", vocab_size=16000,
            character_coverage=1.0, model_type="bpe", pad_id=3
        )
    if not os.path.exists(en_model_file):
        print("Training English SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input="./kftt-data-1.0/data/tok/kyoto-train.en",
            model_prefix="en_spm", vocab_size=16000,
            character_coverage=1.0, model_type="bpe", pad_id=3
        )

    sp_ja = spm.SentencePieceProcessor(model_file=ja_model_file)
    sp_en = spm.SentencePieceProcessor(model_file=en_model_file)

    with open("./kftt-data-1.0/data/tok/kyoto-train.ja", "r", encoding="utf-8") as f:
        train_ja = f.readlines()
    with open("./kftt-data-1.0/data/tok/kyoto-train.en", "r", encoding="utf-8") as f:
        train_en = f.readlines()

    ja_tokenized = [sp_ja.encode_as_ids(line.strip()) for line in train_ja]
    en_tokenized = [sp_en.encode_as_ids(line.strip()) for line in train_en]

    ja_ids = [[sp_ja.bos_id()] + ids + [sp_ja.eos_id()] for ids in ja_tokenized]
    en_ids = [[sp_en.bos_id()] + ids + [sp_en.eos_id()] for ids in en_tokenized]

    # --- DDPによる学習の実行 ---
    world_size = torch.cuda.device_count()
     if world_size > 1:
        args = (world_size, ja_ids, en_ids, sp_ja, sp_en)
        mp.spawn(worker,
                 args=args,
                 nprocs=world_size,
                 join=True)
    else:
        print("DDP requires multiple GPUs. Running on a single GPU is not supported by this script.")
    
    # --- 評価はメインプロセスのみが実行 ---
    print("DDP training finished. Starting evaluation on the main process.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    src_vocab_size = sp_ja.get_piece_size()
    tgt_vocab_size = sp_en.get_piece_size()

    # DDPでラップされていない単体のモデルをインスタンス化
    eval_model = TransformerNMT(src_vocab_size, tgt_vocab_size).to(device)
    eval_model.load_state_dict(torch.load("subword_transformer_nmt_ddp.pt", map_location=device))
    
    # 評価とプロットを実行
    evaluate_and_plot(eval_model, device, sp_ja, sp_en)

if __name__ == '__main__':
    main()
