# model_def.py
import torch
import torch.nn as nn
import pickle
import MeCab

# GPUが利用可能か確認し、デバイスを設定します
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Transformerモデルの定義 (この部分は変更ありません) ---
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

# --- 修正部分 ---
# モデル、辞書、MeCab Taggerを一度に読み込む関数
def load_model_and_dict():
    # MeCab Taggerをここで一度だけ初期化します
    # 注意: この辞書のパスはあなたの環境に合わせてください
    try:
        tagger = MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    except RuntimeError as e:
        print(f"Error initializing MeCab Tagger: {e}")
        print("Please check the dictionary path in model_def.py")
        # MeCabの初期化に失敗した場合は、プログラムを終了させます
        raise

    # 辞書ファイルを読み込みます
    with open("model/ja_token2id.pkl", "rb") as f:
        ja_token2id = pickle.load(f)
    with open("model/en_token2id.pkl", "rb") as f:
        en_token2id = pickle.load(f)
    with open("model/en_id2token.pkl", "rb") as f:
        en_id2token = pickle.load(f)
    
    src_vocab_size = len(ja_token2id)
    tgt_vocab_size = len(en_token2id)

    # モデルのインスタンスを作成します
    model = TransformerNMT(
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_ff=2048
    ).to(device)
    
    # 学習済みモデルの重みを読み込みます
    model.load_state_dict(torch.load("model/model.pt", map_location=device))
    model.eval() # モデルを推論モードに設定します

    # 初期化したオブジェクトを全て返します
    return model, ja_token2id, en_token2id, en_id2token, tagger

# 翻訳処理を行う関数
def translate(model, ja_token2id, en_token2id, en_id2token, tagger, sentence, max_len=50):
    # MeCabの初期化処理を削除し、引数で受け取ったtaggerを使用します
    node = tagger.parseToNode(sentence.strip())
    tokens = []
    while node:
        if node.surface:
            tokens.append(node.surface)
        node = node.next
    
    # トークンをIDに変換し、PyTorchのテンソルに変換します
    src = torch.tensor([[ja_token2id["<sos>"]] + [ja_token2id.get(t, ja_token2id["<unk>"]) for t in tokens] + [ja_token2id["<eos>"]]], device=device)
    src_padding_mask = (src == ja_token2id["<pad>"]).to(device)
    
    generated_ids = [en_token2id["<sos>"]]

    model.eval() # 推論モードであることを確認
    with torch.no_grad(): # 勾配計算を無効化してメモリ効率を上げます
        for _ in range(max_len):
            tgt_input = torch.tensor([generated_ids], device=device)
            tgt_len = tgt_input.size(1)
            # Decoderへの入力マスクを作成します
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

            # モデルで予測を実行します
            output = model(src, tgt_input, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask)
            
            # 最後の単語の予測確率が最も高いものを次の単語とします
            next_token_id = output[0, -1].argmax(-1).item()
            
            # 終了トークンが出たら翻訳を終了します
            if next_token_id == en_token2id["<eos>"]:
                break
            
            generated_ids.append(next_token_id)

    # IDのリストを単語の文字列に変換します
    return " ".join([en_id2token[i] for i in generated_ids[1:]])
