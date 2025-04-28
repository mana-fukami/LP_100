#制御フローを読みやすくする
#巨大な式を分割する

#1行に1記事の情報がJSON形式で格納される
#各行には記事名が”title”キーに、記事本文が”text”キーの辞書オブジェクトに格納
#ファイル全体はgzipで圧縮される

#Wikipedia記事のJSONファイルを読み込み、「イギリス」に関する記事本文を表示せよ。
# 問題21-29では、ここで抽出した記事本文に対して実行せよ。

import json

json_file=open("3\jawiki-country.json","r",encoding="utf-8")
json_lines=json_file.readlines()
articles=[]
for line in json_lines:
    articles.append(json.loads(line))

for article in articles:
    if article["title"]=="イギリス":
        UK_text=article["text"]
        break

print(UK_text)

#実行結果(一部抜粋)
"""
~~~~~
{{EU|1973年 - 2020年}}
{{CPLP}}
{{デフォルトソート:いきりす}}
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]
"""