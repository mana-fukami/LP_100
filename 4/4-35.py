#「メロスは激怒した。」の係り受け木を可視化せよ。

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

import spacy
from spacy import displacy
import ginza

nlp=spacy.load("ja_ginza")
doc=nlp(text)

html=displacy.render(doc,style="dep")
with open("dependency_tree.html", "w", encoding="utf-8") as f:
    f.write(html)

#実行結果：dependency_tree.htmlをlive serverで開くと確認できる。