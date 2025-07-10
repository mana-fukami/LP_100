"""
91で学習したニューラル機械翻訳モデルで翻訳文をデコードする際に,ビーム探索を導入せよ.
ビーム幅を1から100くらいまで適当に変化させながら,開発セット上のBLEUスコアの変化をプロットせよ.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu
import pickle
import matplotlib.pyplot as plt
from heapq import heappush, heappop

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
    tokens = src_sentence.strip().split()
    ids = [ja_token2id.get(t, ja_token2id['<unk>']) for t in tokens]
    src = torch.tensor([[ja_token2id['<sos>']] + ids + [ja_token2id['<eos>']]], device=device)
    src_padding_mask = (src == ja_token2id['<pad>'])

    # ビームノードの初期化
    beams = [BeamNode([en_token2id['<sos>']], 0.0)]
    completed=[BeamNode([en_token2id['<sos>']], 0.0)]

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
                    tgt_padding_mask=(tgt_input==en_token2id["<pad>"]),
                    memory_key_padding_mask=src_padding_mask
                )
            # 出力の最後のトークンの分布
            log_probs = F.log_softmax(out, dim=-1)
            # 上位k個の候補を選ぶ
            topk_log_probs, topk_indices = log_probs.topk(beam_width)

            # 新しいビームノードを生成
            for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                new_seq = beam.seq + [idx.item()]
                new_logprob = beam.logprob + log_prob.item()
                # <eos>トークンが生成された場合は完了候補として別保存
                if idx.item() == en_token2id['<eos>']:
                    completed.append(BeamNode(new_seq, new_logprob))
                new_beams.append(BeamNode(new_seq, new_logprob))

        # 上位k個を再選択し、次の候補にする
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x.logprob)
        if all(beam.seq[-1] == en_token2id['<eos>'] for beam in beams):
            break

    # 最も確率の高い完了候補orビーム候補を選択
    if completed:
        best = max(completed, key=lambda x: x.logprob)
    else:
        best = max(beams, key=lambda x: x.logprob)
    return ' '.join(en_id2token[idx] for idx in best.seq[1:-1])  # remove <sos> and <eos>

# 開発データ読み込み
with open("./kftt-data-1.0/data/tok/kyoto-dev.ja") as f:
    dev_ja = [line.strip().split() for line in f]
with open("./kftt-data-1.0/data/tok/kyoto-dev.en") as f:
    dev_en = [line.strip() for line in f]

beam_widths=list(range(1, 101,10))
bleu_scores = []

# BLEU計算
for width in beam_widths:
    print(f"Beam width: {width}")
    translations = [beam_search_translate(sent, beam_width=width) for sent in dev_ja]
    bleu = sacrebleu.corpus_bleu(translations, [dev_en])
    print(f"BLEU: {bleu.score:.2f}")
    bleu_scores.append(bleu.score)

# 結果のプロット
plt.plot(beam_widths, bleu_scores, marker='o')
plt.xlabel('Beam Width')
plt.ylabel('BLEU Score')
plt.title('Beam Width vs BLEU Score')
plt.grid(True)
plt.show()