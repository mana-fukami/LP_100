#記事中でカテゴリ名を宣言している行を抽出せよ。
#カテゴリーの形式
# \n[[Category:エジプト|*]]\n[[Category:共和国]]\n[[Category:軍事政権]]

import json
import re

json_file=open("jawiki-country.json","r",encoding="utf-8")
json_lines=json_file.readlines()
articles=[]
for line in json_lines:
    articles.append(json.loads(line))

for article in articles:
    if article["title"]=="イギリス":
        UK_text=article["text"]
        break

categories=[]
categories=re.findall("\[\[Category:.*?\]\]",UK_text)

print(categories)