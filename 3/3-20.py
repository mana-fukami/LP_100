#1行に1記事の情報がJSON形式で格納される
#各行には記事名が”title”キーに、記事本文が”text”キーの辞書オブジェクトに格納
#ファイル全体はgzipで圧縮される

#Wikipedia記事のJSONファイルを読み込み、「イギリス」に関する記事本文を表示せよ。
# 問題21-29では、ここで抽出した記事本文に対して実行せよ。

import json

json_file=open("jawiki-country.json","r",encoding="utf-8")
json_lines=json_file.readlines()
articles=[]
for line in json_lines:
    articles.append(json.loads(line))

for article in articles:
    if article["title"]=="イギリス":
        UK_text=article["text"]
        break

print(UK_text)