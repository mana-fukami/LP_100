#記事から参照されているメディアファイルをすべて抜き出せ。
#メディアファイルの形式
# \n[[ファイル:Jan Długosz.PNG|thumb|right|150px|ヤン・ドゥウゴシュ]]
# \n[[ファイル:Jan Kochanowski.png|thumb|right|150px|ヤン・コハノフスキ]]

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

media_files=re.findall("\[\[\ファイル:(.*?)(?:\|.*?)]\]",UK_text)
print(media_files)
