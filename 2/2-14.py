#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#ファイルの先頭10行に対して、各行の1列目だけを抜き出して表示せよ。
# 確認にはcutコマンドなどを用いよ。

original_file=open("2/popular-names.txt","r")
lines=original_file.readlines()

for i in range(10):
    lines[i]=lines[i].replace("\t"," ")
    row=lines[i].split(" ")
    print(row[0])

# 実行結果
"""
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah
"""

# 確認コマンド# head -10 'popular-names.txt' | sed 's/\t/ /g' |cut -f 1 -d " "
"""
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah
"""