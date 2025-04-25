#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#ファイルの先頭N行だけを表示せよ。例えば、N=10として先頭10行を表示せよ。
# 確認にはheadコマンドを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()

N=10
for i in range(N):
    print(lines[i],end="") #lines[i]にも改行は含まれる&print()でも末尾に改行がある→lines[i][1:-1]で改行を削除して表示。

# 実行結果
"""
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880
Margaret        F       1578    1880
Ida     F       1472    1880
Alice   F       1414    1880
Bertha  F       1320    1880
Sarah   F       1288    1880
"""

# 確認コマンド# head -10 'popular-names.txt'
# 実行結果
"""
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880
Margaret        F       1578    1880
Ida     F       1472    1880
Alice   F       1414    1880
Bertha  F       1320    1880
Sarah   F       1288    1880
"""