#文章textにおいて、2つの名詞が「の」で連結されている名詞句をすべて抽出せよ。

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

import MeCab
import unidic
tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir") #辞書の指定
node=tagger.parseToNode(text)
#ノードの構成：
#   node.surface - 表層形
#   node.feature - 品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3, 活用型, 活用形, 原形, 読み,発音
#   node.next - 次のノード
#   node.prev - 前のノード

surface_feature0=[]
while node:
    node_feature=node.feature.split(",")
    surface_feature0.append((node.surface,node_feature[0]))
    node=node.next

pair_nouns=[]
for i in range(1,len(surface_feature0)-1,1):
    prev=surface_feature0[i-1]
    prev_id=i-1
    list=surface_feature0[i]
    next=surface_feature0[i+1]
    next_id=i+1
    if list[0]=="の" and prev[1]=="名詞" and next[1]=="名詞": #名詞句を作る「の」を見つけたら
        pair_noun=""
        while surface_feature0[prev_id-1][1]=="名詞": #名詞句の始まりをprev_idに
            prev_id=prev_id-1
        while surface_feature0[next_id+1][1]=="名詞": #名詞句の終わりをnext_idに
            next_id=next_id+1
        for i in range(prev_id,next_id+1,1):
            pair_noun=pair_noun+str(surface_feature0[i][0])
        pair_nouns.append(str(pair_noun))

print(pair_nouns)
for pair_noun in pair_nouns:
    print(pair_noun)