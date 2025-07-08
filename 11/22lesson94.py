"""
91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に,ビーム探索を導入せよ.
ビーム幅を1から100くらいまで適当に変化させながら,開発セット上のBLEUスコアの変化をプロットせよ.
"""
import torch
import torch.nn as nn
import sacrebleu
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# モデルの定義（91と同じ）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=8, num_layers=4, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src)).transpose(0,1)
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt)).transpose(0,1)
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_layer(output.transpose(0,1))

# 翻訳関数
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def beam_search_translate(model,src_tensor, beam_size=5, max_len=50):
    model.eval()
    with torch.no_grad():
        src_padding_mask=(src_tensor == ja_token2id["<pad>"])
        memory=model.transformer.encoder(
            model.positional_encoding(model.src_embedding(src_tensor)).transpose(0,1),
            src_key_padding_mask=src_padding_mask
        )
        beams=[(torch.tensor([en_token2id["<sos>"]], device=device), 0.0)]
        for _ in range(max_len):
            new_beams=[]
            for seq, score in beams:
                if seq[-1].item() == en_token2id["<eos>"]:
                    new_beams.append((seq, score))
                    continue
                tgt_input = seq.unsqueeze(0)
                tgt_mask = generate_square_subsequent_mask(tgt_input.size(1), device)
                out = model.transformer.decoder(
                    model.positional_encoding(model.tgt_embedding(tgt_input)).transpose(0, 1),
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                logits = model.output_layer(out.transpose(0, 1))[:, -1, :]  # [1, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                for i in range(beam_size):
                    next_seq = torch.cat([seq, topk_indices[i].view(1)])
                    next_score = score + topk_log_probs[i].item()
                    new_beams.append((next_seq, next_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        return beams[0][0]


# 語彙読み込み
import pickle
with open("ja_token2id.pkl", "rb") as f:
    ja_token2id = pickle.load(f)
with open("en_token2id.pkl", "rb") as f:
    en_token2id = pickle.load(f)
with open("en_id2token.pkl", "rb") as f:
    en_id2token = pickle.load(f)

src_vocab_size = len(ja_token2id)
tgt_vocab_size = len(en_token2id)

# モデル読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerNMT(
    src_vocab_size,
    tgt_vocab_size,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_ff=2048
).to(device)
model.load_state_dict(torch.load("transformer_nmt.pt"))
model.eval()

# テストデータ読み込み
with open("./kftt-data-1.0/data/tok/kyoto-dev.ja") as f:
    dev_ja = [line.strip().split() for line in f]
with open("./kftt-data-1.0/data/tok/kyoto-dev.en") as f:
    dev_en = [line.strip() for line in f]

beam_widths=list(range(1, 101,10))
bleu_scores = []

# BLEU計算
for beam in beam_widths:
    print(f"Beam width: {beam}")
    preds = []
    refs = []
    for ja_sent, en_sent in zip(dev_ja[:100], dev_en[:100]):
        out_tokens = beam_search_translate(model, ja_sent, ja_token2id, en_token2id, en_id2token, beam_width=beam, device='cuda')
        pred_sent = ' '.join(out_tokens)
        preds.append(pred_sent)
        refs.append([en_sent])
    bleu = sacrebleu.corpus_bleu(preds, refs).score
    print(f"BLEU = {bleu:.2f}")
    bleu_scores.append(bleu)

# 結果のプロット
plt.plot(beam_widths, bleu_scores, marker='o')
plt.xlabel('Beam Width')
plt.ylabel('BLEU Score')
plt.title('Beam Width vs BLEU Score')
plt.grid(True)
plt.show()