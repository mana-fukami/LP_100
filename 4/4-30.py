#文章textに含まれる動詞をすべて表示せよ。

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

#形態素解析に使うライブラリをインポートする
import MeCab
import UniDic
tagger=MeCab.Tagger()
nodes=tagger.parseToNode(text) #各形態素をノードとして取得する#ノード＝構造体リスト
#ノードの構成：
#   node.surface - 表層形
#   node.feature - 品詞、活用、原型をカンマ区切りで並べた文字列
#   node.next - 次のノード
#   node.stat - BOS/EOSなどのステータス
#   node.length - 形態素の文字数
#   node.rlength - 表示される長さ

verb=[]
while nodes:
    if nodes.surface != "":
        node_material=nodes.feature.split(",")
        if node_material[0]=="動詞":
            verb.append(nodes.surface)
    nodes=nodes.next

print(verb)