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
import unidic
#rを文字列前につけると、\を特殊文字として扱わない
#rをつけないと、\は特殊文字として扱われるため、処理が止まる
tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir") #辞書の指定
node=tagger.parseToNode(text) #各形態素をノードとして取得する#ノード＝構造体リスト
#ノードの構成：
#   node.surface - 表層形
#   node.feature - 品詞、活用、原型をカンマ区切りで並べた文字列
#   node.next - 次のノード
#   node.stat - BOS/EOSなどのステータス
#   node.length - 形態素の文字数
#   node.rlength - 表示される長さ

verb=[]
while node:
    if node.surface != "":
        node_material=node.feature.split(",")
        if node_material[0]=="動詞":
            verb.append(node.surface)
    node=node.next

print(verb)
# 実行結果
"""
['し', '除か', 'なら', 'し', 'わから', 'ある', '吹き', '遊ん', '暮し', '来', '対し', 'あっ']
"""