#文章textに係り受け解析を適用し、係り元と係り先のトークン
# （形態素や文節などの単位）をタブ区切り形式ですべて抽出せよ。

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

print('---- bunsetu_spans ----')
for span in ginza.bunsetu_spans(doc):
    for token in span.lefts:
        print(f'{token} : {str(ginza.bunsetu_span(token))} → {str(span)}')


print('---- bunsetu_phrase_spans (主辞) ----')
for span in ginza.bunsetu_phrase_spans(doc):
    for token in span.lefts:
        print(f'{token} : {str(ginza.bunsetu_span(token))} → {str(span)}')

