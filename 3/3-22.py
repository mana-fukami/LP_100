#記事のカテゴリ名を（行単位ではなく名前で）抽出せよ
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

category_names=re.findall("\[\[Category:(.*?)(?:\|.*?|)\]\]",UK_text)
#(?:~~)→文字列"~~"を除外する
print(category_names)