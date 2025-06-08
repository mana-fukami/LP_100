"""
複数の事例が与えられたとき、これらをまとめて一つのテンソル・オブジェクトで表現する関数collateを実装せよ。
与えられた複数の事例のトークン列の長さが異なるときは、トークン列の長さが最も長いものに揃え、0番のトークンIDでパディングをせよ。
さらに、トークン列の長さが長いものから順に、事例を並び替えよ。
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import tensor

def collate(batch):
    # トークン列の長さでソートする
    batch.sort(key=lambda x:len(x["input_ids"]),reverse=True)
    labels = torch.stack([item["label"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]

    # input_idsをパディングして同じ長さに揃える
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {"input_ids": padded_input_ids,"label": labels}

batch=[
    {'text': 'hide new secretions from the parental units','label': tensor([0.]),'input_ids': tensor([  5785,     66, 113845,     18,     12,  15095,   1594])},
    {'text': 'contains no wit , only labored gags','label': tensor([0.]),'input_ids': tensor([ 3475,    87, 15888,    90, 27695, 42637])},
    {'text': 'that loves its characters and communicates something rather beautiful about human nature','label': tensor([1.]),'input_ids': tensor([    4,  5053,    45,  3305, 31647,   348,   904,  2815,    47,  1276,  1964])},
    {'text': 'remains utterly satisfied to remain the same throughout','label': tensor([0.]),'input_ids': tensor([  987, 14528,  4941,   873,    12,   208,   898])}
    ]

result = collate(batch)
print(result)
# 実行例
"""
{'input_ids': tensor([
        [     4,   5053,     45,   3305,  31647,    348,    904,   2815,     47,   1276,   1964],
        [  5785,     66, 113845,     18,     12,  15095,   1594,      0,      0,      0,      0],
        [   987,  14528,   4941,    873,     12,    208,    898,      0,      0,      0,      0],
        [  3475,     87,  15888,     90,  27695,  42637,      0,      0,      0,      0,      0]]),
'label': tensor([[1.],
        [0.],
        [0.],
        [0.]])}
"""