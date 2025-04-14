#26の処理に加えて、テンプレートの値からMediaWikiの内部リンクマークアップを除去し、
# テキストに変換せよ（参考: マークアップ早見表）。

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
    not_bold_info=re.sub("'*?","",info[1])
    info_no_internal_link=re.sub("\[\[.*\|","",not_bold_info)
    info_no_internal_link=re.sub("\[\[","",info_no_internal_link)
    info_no_internal_link=re.sub("\]\]","",info_no_internal_link)
    info_template[info[0]]=info_no_internal_link

#print(UK_text)
#print(fundamental_info)
print(info_template)