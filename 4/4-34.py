#文章textにおいて、「メロス」が主語であるときの述語を抽出せよ。

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

import spacy
import ginza

nlp=spacy.load("ja_ginza")
doc=nlp(text)

for span in ginza.bunsetu_spans(doc): #係り付け結果が文節で返される
    for token in span.lefts:
        if "メロス" in str(ginza.bunsetu_span(token)):
            print(str(ginza.bunsetu_span(token))+"  "+str(span))

#実行結果
"""
メロスは  激怒した。

メロスには  わからぬ。

メロスは、  牧人である。
"""