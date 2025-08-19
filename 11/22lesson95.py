"""
トークンの単位を単語や形態素からサブワードに変更し，91-94の実験を再度実施せよ．
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import math
import sacrebleu
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch.nn.functional as F
import heapq
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# GPUに移動する
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# サブワード単位に分割できるように、SentencePieceを学習する
# -学習データ読み込み
with open("./kftt-data-1.0/data/tok/kyoto-train.ja","r",encoding="utf-8") as f:
    train_ja = f.readlines()
with open("./kftt-data-1.0/data/tok/kyoto-train.en","r",encoding="utf-8") as f:
    train_en = f.readlines()
# -日本語用モデル
spm.SentencePieceTrainer.train(
    input="./kftt-data-1.0/data/tok/kyoto-train.ja",
    model_prefix="ja_spm",
    vocab_size=16000,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=3
)
# -英語用モデル
spm.SentencePieceTrainer.train(
    input="./kftt-data-1.0/data/tok/kyoto-train.en",
    model_prefix="en_spm",
    vocab_size=16000,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=3
)
# -トークン化されたデータの再生成
sp_ja= spm.SentencePieceProcessor(model_file="ja_spm.model")
sp_en= spm.SentencePieceProcessor(model_file="en_spm.model")

ja_tokenized= [sp_ja.encode_as_ids(line.strip()) for line in train_ja]
en_tokenized= [sp_en.encode_as_ids(line.strip()) for line in train_en]

# -数値列に変換
ja_ids = [[sp_ja.bos_id()] + ids + [sp_ja.eos_id()] for ids in ja_tokenized]
en_ids = [[sp_en.bos_id()] + ids + [sp_en.eos_id()] for ids in en_tokenized]

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
    src_batch = pad_sequence(src_batch, padding_value=sp_ja.pad_id(), batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=sp_en.pad_id(), batch_first=True)
    return src_batch, tgt_batch

train_dataset=TranslationDataset(ja_ids,en_ids)
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=Collate)

#モデルの学習
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
src_vocab_size = max(
    max([max(ids) for ids in ja_ids]),
    sp_ja.bos_id(),
    sp_ja.eos_id(),
    sp_ja.pad_id(),
    sp_ja.unk_id()
) + 1
tgt_vocab_size = max(
    max([max(ids) for ids in en_ids]),
    sp_en.bos_id(),
    sp_en.eos_id(),
    sp_en.pad_id(),
    sp_en.unk_id()
) + 1
print("---- ID CHECK ----")
print(f"sp_ja vocab size: {sp_ja.get_piece_size()}")
print(f"sp_ja <bos>: {sp_ja.bos_id()}, <eos>: {sp_ja.eos_id()}, <pad>: {sp_ja.pad_id()}, <unk>: {sp_ja.unk_id()}")
print(f"Max src ID in ja_ids: {max([max(ids) for ids in ja_ids])}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNMT(
    src_vocab_size,
    tgt_vocab_size,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_ff=2048
).to(device)
model = nn.DataParallel(model)  # DataParallelでラップ
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9,0.98), eps=1e-9)
pad_id = sp_en.pad_id()
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

        src_pad_id = sp_ja.pad_id()
        tgt_pad_id = sp_en.pad_id()

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

# モデルの保存
torch.save(model.state_dict(), "subword_transformer_nmt.pt")

# 翻訳BLEU評価 - greedy search
# テストデータ読み込み
with open("./kftt-data-1.0/data/orig/kyoto-test.ja") as f:
    test_ja = [line.strip() for line in f]
with open("./kftt-data-1.0/data/orig/kyoto-test.en") as f:
    test_en = [line.strip() for line in f]

# 翻訳関数
def translate(sentence, max_len=50):
    tokens=sp_ja.encode_as_ids(sentence)
    src = torch.tensor([[sp_ja.bos_id()] + tokens + [sp_ja.eos_id()]], device=device)
    src_padding_mask = (src == sp_ja.pad_id())
    generated = [sp_en.bos_id()]
    for _ in range(max_len):
        tgt_input = torch.tensor([generated], device=device)
        tgt_mask = torch.triu(torch.ones(tgt_input.size(1), tgt_input.size(1), device=device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
        with torch.no_grad():
            out = model(
                src,
                tgt_input,
                tgt_mask=tgt_mask,
                src_padding_mask=src_padding_mask,
                tgt_padding_mask=(tgt_input==sp_en.pad_id()),
                memory_key_padding_mask=src_padding_mask
            )
        next_token = out[0, -1].argmax(-1).item()
        generated.append(next_token)
        if next_token == sp_en.eos_id():
            break
    return sp_en.decode_ids(generated[1:-1])

# BLEU計算
hypotheses = []
for sentence in tqdm(test_ja):
    pred_tokens = translate(sentence)
    hypotheses.append(" ".join(pred_tokens))

# sacreBLEU
bleu = sacrebleu.corpus_bleu(hypotheses, [test_en])
print(f"BLEU: {bleu.score:.2f}")

# 翻訳BLEU - ビームサーチ
# マスク生成
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

# ビームサーチによる翻訳
# --ビームサーチ：上位k個の候補文を並列に保持市ながら進め、各候補について次の語を生成し、上位k個を選びなおす
class BeamNode:
    def __init__(self, seq, logprob):
        self.seq = seq # トークン列
        self.logprob = logprob # 対数確率

    def __lt__(self, other):
        # 比較関数を定義して、ソート時に使用
        return self.logprob > other.logprob  # max-heap

def beam_search_translate(src_sentence, beam_width=5, max_len=50):
    tokens = sp_ja.encode_as_ids(src_sentence)
    src = torch.tensor([[sp_ja.bos_id()] + tokens + [sp_ja.eos_id()]], device=device)
    src_padding_mask = (src == sp_ja.pad_id())

    # ビームノードの初期化
    beams = [BeamNode(sp_en.bos_id(), 0.0)]
    completed=[BeamNode(sp_en.bos_id(), 0.0)]

    # ビームサーチのメインループ
    for _ in range(max_len):
        new_beams = []
        # 各候補について次の単語を予測する
        for beam in beams:
            tgt_input = torch.tensor([beam.seq], device=device)
            tgt_mask = generate_square_subsequent_mask(len(beam.seq)).to(device)
            # Transformerに現在のsrcとtgt_inputを入力
            with torch.no_grad():
                out = model(
                    src,
                    tgt_input,
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=(tgt_input==sp_en.pad_id()),
                    memory_key_padding_mask=src_padding_mask
                )
            # 出力の最後のトークンの分布
            log_probs = F.log_softmax(out[:, -1, :], dim=-1)
            # 上位k個の候補を選ぶ
            topk_log_probs, topk_indices = log_probs.topk(beam_width)

            print(f"topk_indices shape: {topk_indices.shape}")

            # 新しいビームノードを生成
            for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                new_seq = beam.seq + [idx.item()]
                new_logprob = beam.logprob + log_prob.item()
                # <eos>トークンが生成された場合は完了候補として別保存
                if idx.item() == sp_en.eos_id():
                    completed.append(BeamNode(new_seq, new_logprob))
                new_beams.append(BeamNode(new_seq, new_logprob))

        # 上位k個を再選択し、次の候補にする
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x.logprob)
        if all(beam.seq[-1] == sp_en.eos_id()for beam in beams):
            break

    # 最も確率の高い完了候補orビーム候補を選択
    if completed:
        best = max(completed, key=lambda x: x.logprob)
    else:
        best = max(beams, key=lambda x: x.logprob)
    return ' '.join(sp_en.decode_ids(best.seq[1:-1]))  # remove <sos> and <eos>

# 開発データ読み込み
with open("./kftt-data-1.0/data/orig/kyoto-dev.ja","r",encoding="utf-8") as f:
    dev_ja = [line.strip() for line in f]
with open("./kftt-data-1.0/data/orig/kyoto-dev.en","r",encoding="utf-8") as f:
    dev_en = [line.strip() for line in f]

beam_widths=list(range(1, 101,10))
bleu_scores = []

# BLEU計算
for width in beam_widths:
    print(f"Beam width: {width}")
    translations = []
    for sent in tqdm(dev_ja):
        translations.append(beam_search_translate(sent, beam_width=width))
    bleu = sacrebleu.corpus_bleu(translations, [dev_en])
    print(f"BLEU: {bleu.score:.2f}")
    bleu_scores.append(bleu.score)

# 結果のプロット
plt.plot(beam_widths, bleu_scores, marker='o')
plt.xlabel('Beam Width')
plt.ylabel('BLEU Score')
plt.title('Beam Width vs BLEU Score')
plt.grid(True)
#plt.show()
plt.savefig('subword_beam_width_vs_bleu.png', dpi=300, bbox_inches='tight')
plt.close()