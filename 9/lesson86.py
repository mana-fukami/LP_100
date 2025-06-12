"""
85で読み込んだ訓練データの一部(例えば冒頭の4事例)に対して、パディングなどの処理を行い、
トークン列の長さを揃えてミニバッチを構成せよ。
"""
from lesson85 import train_dataset
import torch
import numpy as np

def CustomCollate(batch):
    #データを各項目についてまとめる
    labels=torch.stack([item["label"] for item in batch])
    texts=[item["text"] for item in batch]
    tokens=[item["tokens"] for item in batch]
    #パディングを行いトークン列の長さをそろえる
    # -最も長いトークン列の長さを取得する
    max_len=np.max([len(l) for l in tokens])
    padded_tokens=[]
    for t in tokens:
        for i in range(len(t),max_len,1):
            t.append("")
        padded_tokens.append(t)

    return {"text":texts,"label":labels,"tokens":padded_tokens}

batch=train_dataset[:4]
padded_batch=CustomCollate(batch)
def show_result():
    print(padded_batch)
show_result()
# 実行結果
"""
{'text': ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ', 'remains utterly satisfied to remain the same throughout '], 'label': tensor([0., 0., 1., 0.]), 'tokens': [['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '', '', '', '', ''], ['contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s', '', '', '', ''], ['that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature'], ['remains', 'utterly', 'satisfied', 'to', 'remain', 'the', 'same', 'throughout', '', '', '', '', '']]}
"""
# 見やすく改行を加えた
"""
{'text':[
    'hide new secretions from the parental units ',
    'contains no wit , only labored gags ',
    'that loves its characters and communicates something rather beautiful about human nature ',
    'remains utterly satisfied to remain the same throughout '
    ],
'label': tensor([0., 0., 1., 0.]),
'tokens': [
    ['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '', '', '', '', ''],
    ['contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s', '', '', '', ''],
    ['that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature'],
    ['remains', 'utterly', 'satisfied', 'to', 'remain', 'the', 'same', 'throughout', '', '', '', '', '']]}
"""