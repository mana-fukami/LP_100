#テンプレートの内容を利用し、国旗画像のURLを取得せよ。
# （ヒント: MediaWiki APIのimageinfoを呼び出して、ファイル参照をURLに変換すればよい）

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

#APIリクエスト
import requests

S=requests.Session()
URL="https://en.wikipedia.org/w/api.php"

PARAMS={
    "action":"query",
    "format":"json",
    "prop":"imageinfo",
    "titles":"File:"+info_template["国旗画像"],
    "iiprop":"url"
}

R=S.get(url=URL,params=PARAMS)
DATA=R.json()

PAGES=DATA["query"]["pages"]

#print(UK_text)
#print(fundamental_info)
for k, v in PAGES.items():
    print(v["imageinfo"][0]["url"])