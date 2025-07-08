"""
91で学習したニューラル機械翻訳モデルを用い,
与えられた（任意の）日本語の文を英語に翻訳するプログラムを実装せよ．
"""
import torch
import torch.nn as nn
import MeCab

# 学習時のTransformerNMTと同じ構造を再利用
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


# --- 語彙のロード（訓練時と同じファイルから）
import pickle
with open("ja_token2id.pkl", "rb") as f:
    ja_token2id = pickle.load(f)
with open("en_token2id.pkl", "rb") as f:
    en_token2id = pickle.load(f)
with open("en_id2token.pkl", "rb") as f:
    en_id2token = pickle.load(f)

src_vocab_size = len(ja_token2id)
tgt_vocab_size = len(en_token2id)

# --- モデルの復元
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

# --- 翻訳関数
def translate(sentence):
    # トークナイズする
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(sentence)
    tokens = []
    while node:
        if node.surface != "":
            tokens.append(node.surface)
        node = node.next
    ids = [ja_token2id.get(token, ja_token2id["<unk>"]) for token in tokens]
    src = torch.tensor([[ja_token2id["<sos>"]] + ids + [ja_token2id["<eos>"]]], device=device)
    
    src_padding_mask = (src == ja_token2id["<pad>"])
    
    # デコーダの初期入力
    generated = [en_token2id["<sos>"]]
    
    max_len = 50
    for i in range(max_len):
        tgt_input = torch.tensor([generated], device=device)
        tgt_mask = torch.triu(torch.ones(tgt_input.size(1), tgt_input.size(1), device=device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
        
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
    # ID列 → 単語列
    output_tokens = [en_id2token[idx] for idx in generated[1:-1]]  # <sos> と <eos> を除外
    return " ".join(output_tokens)

# --- 実行例
while True:
    jp_sentence = input("日本語文> ")
    if jp_sentence == "exit":
        break
    translation = translate(jp_sentence)
    print("翻訳結果:", translation)

"""
日本語文> こんにちは
翻訳結果: <unk>
日本語文> 今日は晴れです
翻訳結果: Today , the <unk> ( a <unk> ) is <unk> .
日本語文> 彼女は私の友人です
翻訳結果: She was a friend of her .
日本語文> 夏よりも春が好きです
翻訳結果: In summer , spring is also used for spring and spring .
日本語文> 私は夏よりも春のほうが好きです
翻訳結果: I have a good reputation for summer , but I have a good reputation than spring .
日本語文> 
"""