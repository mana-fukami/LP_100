#27の処理に加えて、テンプレートの値からMediaWikiマークアップを可能な限り除去し、
# 国の基本情報を整形せよ。

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
    new_str=re.sub("'*?","",info[1])
    new_str=re.sub("\{\{.*?\|.*?\|","",new_str)
    new_str=re.sub("\{\{.*?\|","",new_str)
    new_str=re.sub("\{\{","",new_str)
    new_str=re.sub("\}\}","",new_str)
    new_str=re.sub("\[\[ファイル:.*?\|.*?\|","",new_str)
    new_str=re.sub("\[\[([^\[])*?\|([^\[])*?\|","",new_str)
    new_str=re.sub("\[\[([^\[])*?\|","",new_str)
    new_str=re.sub("\[\[","",new_str)
    new_str=re.sub("\]\]","",new_str)
    new_str=re.sub("\<ref.*?\>.*?</ref>","",new_str)
    new_str=re.sub("\<.*?/\>","",new_str)
    info_template[info[0]]=new_str

#print(UK_text)
#print(fundamental_info)
print(info_template)