#記事中に含まれるセクション名とそのレベル
# （例えば”== セクション名 ==”なら1）を表示せよ。
#セクションの形式
# ===*===　セクション＝2
# (=の数)‐1=(セクションレベル)

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

sections=re.findall("===*.*?=*==",UK_text)
section_name_level={} #[name:level]

for section in sections:
    level=int(section.count("=")/2)-1
    name=re.findall("===*(.*?)=*==",section)
    section_name_level[name[0]]=level

#print(sections)
print(section_name_level)
