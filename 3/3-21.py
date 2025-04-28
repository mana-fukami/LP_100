#制御フローを読みやすくする
#巨大な式を分割する

#記事中でカテゴリ名を宣言している行を抽出せよ。
#カテゴリーの形式
# \n[[Category:エジプト|*]]\n[[Category:共和国]]\n[[Category:軍事政権]]

import json
import re

json_file=open("3\jawiki-country.json","r",encoding="utf-8")
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

#実行結果
"""
['[[Category:イギリス|*]]', '[[Category:イギリス連邦加盟国]]', '[[Category:英連邦王国|*]]', '[[Category:G8加盟国]]', '[[Category:欧州連合加盟国|元]]', '[[Category:海洋国家]]', '[[Category:現存する君主国]]', '[[Category:島国]]', '[[Category:1801年に成立した国家・領域]]']
"""