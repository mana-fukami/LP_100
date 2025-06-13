"""
85で読み込んだ訓練データの一部(例えば冒頭の4事例)に対して、パディングなどの処理を行い、
トークン列の長さを揃えてミニバッチを構成せよ。
"""
from lesson85 import train_dataset,tokenizer
import torch
import numpy as np

def CustomCollate(batch):
    #データを各項目についてまとめる
    labels=torch.stack([item["label"] for item in batch])
    texts=[item["text"] for item in batch]
    tokens=[item["tokens"] for item in batch]
    #テキストをパディングありでトークナイズする
    padded_tokens=tokenizer(texts,padding=True,return_tensors="pt")

    return {"text":texts,"label":labels,"tokens":padded_tokens}

def show_result():
    batch=train_dataset[:4]
    padded_batch=CustomCollate(batch)
    print(padded_batch)
#show_result()
# 実行結果
"""
{'text': ['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ', 'remains utterly satisfied to remain the same throughout '], 'label': tensor([0., 0., 1., 0.]), 'tokens': {'input_ids': tensor([[  101,  5342,  2047,  3595,  8496,  2013,  1996, 18643,  3197,   102,
             0,     0,     0,     0,     0],
        [  101,  3397,  2053, 15966,  1010,  2069,  4450,  2098, 18201,  2015,
           102,     0,     0,     0,     0],
        [  101,  2008,  7459,  2049,  3494,  1998, 10639,  2015,  2242,  2738,
          3376,  2055,  2529,  3267,   102],
        [  101,  3464, 12580,  8510,  2000,  3961,  1996,  2168,  2802,   102,
             0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])}}
"""
# 見やすく整えた
"""
{
    'text':['hide new secretions from the parental units ', 'contains no wit , only labored gags ', 'that loves its characters and communicates something rather beautiful about human nature ', 'remains utterly satisfied to remain the same throughout '],
    'label': tensor([0., 0., 1., 0.]),
    'tokens': {
        'input_ids':
            tensor([
                [  101,  5342,  2047,  3595,  8496,  2013,  1996, 18643,  3197,   102,     0,     0,     0,     0,     0],
                [  101,  3397,  2053, 15966,  1010,  2069,  4450,  2098, 18201,  2015,   102,     0,     0,     0,     0],
                [  101,  2008,  7459,  2049,  3494,  1998, 10639,  2015,  2242,  2738,  3376,  2055,  2529,  3267,   102],
                [  101,  3464, 12580,  8510,  2000,  3961,  1996,  2168,  2802,   102,     0,     0,     0,     0,     0]
            ]),
        'token_type_ids':
            tensor([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]),
        'attention_mask':
            tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
            ])
    }
}
"""