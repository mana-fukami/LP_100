#制御フローを読みやすくする
#巨大な式を分割する
#コメントすべき事を知る
#コメントは正確で簡潔に

#27の処理に加えて、テンプレートの値からMediaWikiマークアップを可能な限り除去し、
# 国の基本情報を整形せよ。

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

fundamental_info=re.findall("\|(.*?) = (.*?)\n",UK_text)

info_template={}
for info in fundamental_info:
    #強調マークアップの除去
    new_str=re.sub("'*?","",info[1])
    #{}関連のマークアップの除去
    new_str=re.sub("\{\{.*?\|.*?\|","",new_str)
    new_str=re.sub("\{\{.*?\|","",new_str)
    new_str=re.sub("\{\{","",new_str)
    new_str=re.sub("\}\}","",new_str)
    #[]関連のマークアップの除去
    new_str=re.sub("\[\[ファイル:.*?\|.*?\|","",new_str)
    new_str=re.sub("\[\[([^\[])*?\|([^\[])*?\|","",new_str)
    new_str=re.sub("\[\[([^\[])*?\|","",new_str)
    new_str=re.sub("\[\[","",new_str)
    new_str=re.sub("\]\]","",new_str)
    #<>~</>関連のマークアップの除去
    new_str=re.sub("\<ref.*?\>.*?</ref>","",new_str)
    new_str=re.sub("\<.*?/\>","",new_str)
    info_template[info[0]]=new_str

#print(UK_text)
#print(fundamental_info)
print(info_template)

#実行結果
"""
{'略名': 'イギリス', '日本語国名': 'グレートブリテン及び北アイルランド連合王国', '公式国名': 'United Kingdom of Great Britain and Northern Ireland', '国旗画像': 'Flag of the United Kingdom.svg', '国章画像': 'イギリスの国章', '標語': 'Dieu et mon droit（フランス語:神と我が権利）', '国歌': 'God Save the Queen ファイル:United States Navy Band - God Save the Queen.ogg', '地図画像': 'Europe-UK.svg', '位置画像': 'United Kingdom (+overseas territories) in the World (+Antarctica claims).svg', '公用語': '英語', '首都': 'ロンドン（事実上）', '最大都市': 'ロンドン', '元首等肩書': '女王', '元首等氏名': 'エリザベス2世', '首相等肩書': '首相', '首相等氏名': 'ボリス・ジョンソン', '他元首等肩書1': '貴族院議長', '他元首等氏名1': 'ノーマン・ファウラー', '他元首等肩書2': '庶民院議長', '他元首等氏名2': 'en|Lindsay Hoyle', '他元首等肩書3': '最高裁判所長官', '他元首等氏名3': 'ブレンダ・ヘイル', '面積順位': '76', '面積大きさ': '1 E11', '面積値': '244,820', '水面積率': '1.3%', '人口統計年': '2018', '人口順位': '22', '人口大きさ': '1 E7', '人口値': '6643万5600', '人口密度値': '271', 'GDP統計年元': '2012', 'GDP値元': '1兆5478億', 'GDP統計年MER': '2012', 'GDP順位MER': '6', 'GDP値MER': '2兆4337億', 'GDP統計年': '2012', 'GDP順位': '6', 'GDP値': '2兆3162億', 'GDP/人': '36,727', '建国形態': '建国', '確立形態1': 'イングランド王国／スコットランド王国 （両国とも1707年合同法まで）', '確立年月日1': '927年／843年', '確立形態2': 'グレートブリテン王国成立（1707年合同法）', '確立年月日2': '1707年05月01日', '確立形態3': 'グレートブリテン及びアイルランド連合王国成立（1800年合同法）', '確立年月日3': '1801年01月01日', '確立形態4': '現在の国号「グレートブリテン及び 北アイルランド連合王国」に変更', '確立年月日4': '1927年04月12日', '通貨': 'UKポンド (£)', '通貨コード': 'GBP', '時間帯': '±0', '夏時間': '+1', 'ISO 3166-1': 'GB / GBR', 'ccTLD': '.uk / .gb', '国際電話番号': '44', '注記': ''}
"""