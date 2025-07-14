# model_def.py
import torch
import torch.nn as nn
import pickle
import math
import MeCab
# GPU移動
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Transformerモデルの定義
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

# モデルと辞書の読み込み
def load_model_and_dict():
    with open("model/ja_token2id.pkl", "rb") as f:
        ja_token2id = pickle.load(f)
    with open("model/en_token2id.pkl", "rb") as f:
        en_token2id = pickle.load(f)
    with open("model/en_id2token.pkl", "rb") as f:
        en_id2token = pickle.load(f)
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
    model.load_state_dict(torch.load("model/model.pt", map_location="cpu"))
    model.eval()

    return model, ja_token2id, en_token2id, en_id2token

# 翻訳処理
def translate(model, ja_token2id, en_token2id, en_id2token, sentence, max_len=50):
    # トークナイズする
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(sentence.strip())
    tokens = []
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    src = torch.tensor([[ja_token2id.get("<sos>")] + [ja_token2id.get(t, ja_token2id["<unk>"]) for t in tokens] + [ja_token2id.get("<eos>")]])
    src_mask = (src == ja_token2id["<pad>"])
    generated = [en_token2id["<sos>"]]

    for _ in range(max_len):
        tgt_input = torch.tensor([generated])
        tgt_len = tgt_input.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len) == 1).transpose(0, 1).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        out = model(src, tgt_input, tgt_mask=tgt_mask, src_padding_mask=src_mask, memory_key_padding_mask=src_mask)
        next_token = out[0, -1].argmax(-1).item()
        if next_token == en_token2id["<eos>"]:
            break
        generated.append(next_token)

    return " ".join([en_id2token[i] for i in generated[1:]])
