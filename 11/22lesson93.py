"""
91で学習したニューラル機械翻訳モデルの品質を調べるため,
評価データにおけるBLEUスコアを測定せよ.
BLEUスコアの計算にはsacreBLEUを使うこと
"""
import torch
import torch.nn as nn
import sacrebleu
from torch.utils.data import DataLoader
from tqdm import tqdm

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
with open("./kftt-data-1.0/data/tok/kyoto-test.ja") as f:
    test_ja = [line.strip().split() for line in f]
with open("./kftt-data-1.0/data/tok/kyoto-test.en") as f:
    test_en = [line.strip() for line in f]

# 翻訳関数
def translate(tokens, max_len=50):
    ids = [ja_token2id.get(tok, ja_token2id["<unk>"]) for tok in tokens]
    src = torch.tensor([[ja_token2id["<sos>"]] + ids + [ja_token2id["<eos>"]]], device=device)
    src_padding_mask = (src == ja_token2id["<pad>"])
    generated = [en_token2id["<sos>"]]
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
                tgt_padding_mask=(tgt_input==en_token2id["<pad>"]),
                memory_key_padding_mask=src_padding_mask
            )
        next_token = out[0, -1].argmax(-1).item()
        generated.append(next_token)
        if next_token == en_token2id["<eos>"]:
            break
    return [en_id2token[idx] for idx in generated[1:-1]]

# BLEU計算
hypotheses = []
for tokens in tqdm(test_ja):
    pred_tokens = translate(tokens)
    hypotheses.append(" ".join(pred_tokens))

# sacreBLEU
bleu = sacrebleu.corpus_bleu(hypotheses, [test_en])
print(f"BLEU: {bleu.score:.2f}")
"""
100%|██████████████████████████████████████████████████████| 1160/1160 [02:16<00:00,  8.48it/s]
That's 100 lines that end in a tokenized period ('.')
It looks like you forgot to detokenize your test data, which may hurt your score.
If you insist your data is detokenized, or don't care, you can suppress this message with the `force` parameter.
BLEU: 6.57
"""