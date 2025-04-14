#25の処理時に、テンプレートの値からMediaWikiの強調マークアップ
# （弱い強調、強調、強い強調のすべて）を除去してテキストに変換せよ
# （参考: マークアップ早見表）。

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

fundamental_info=re.findall("\|(.*?) = (.*?)\n",UK_text)

info_template={}
for info in fundamental_info:
    info_template[info[0]]=re.sub("'*?","",info[1])

#print(UK_text)
#print(fundamental_info)
print(info_template)