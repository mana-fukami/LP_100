#制御フローを読みやすくする
#文章textに含まれる動詞と、その原型をすべて表示せよ。

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

verb={} # key=原型,value=表層形
while node:
    if node.surface != "":
        node_material=node.feature.split(",")
        if node_material[0]=="動詞":
            #原型はfeatureの7番目に格納されている。
            verb[node_material[6]]=node.surface #原型をkeyに、表層形をvalueにする
    node=node.next

for key,value in verb.items():
    print("表層形："+value,"原型："+key)

"""
実行結果

表層形：し 原型：スル
表層形：除か 原型：ノゾク
表層形：なら 原型：ナル
表層形：わから 原型：ワカル
表層形：あっ 原型：アル
表層形：吹き 原型：フク
表層形：遊ん 原型：アソブ
表層形：暮し 原型：クラス
表層形：来 原型：クル
表層形：対し 原型：タイスル
"""