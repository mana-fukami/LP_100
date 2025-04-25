#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#ファイルの先頭10行に対して、タブ1文字につきスペース1文字に置換して出力せよ。
# 確認にはsedコマンド、trコマンド、もしくはexpandコマンドなどを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()
for i in range(10):
    lines[i]=lines[i].replace("\t"," ") #pythonでtab="\t"
    print(lines[i],end="")

# 実行結果
"""
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880
Margaret F 1578 1880
Ida F 1472 1880
Alice F 1414 1880
Bertha F 1320 1880
Sarah F 1288 1880
"""

# 確認コマンド# head -10 'popular-names.txt' | sed 's/\t/ /g'
# 実行結果
"""
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880
Margaret F 1578 1880
Ida F 1472 1880
Alice F 1414 1880
Bertha F 1320 1880
Sarah F 1288 1880
"""