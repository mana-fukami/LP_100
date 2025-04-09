#ファイルの先頭10行に対して、タブ1文字につきスペース1文字に置換して出力せよ。
# 確認にはsedコマンド、trコマンド、もしくはexpandコマンドなどを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()
for i in range(10):
    lines[i]=lines[i].replace("\t"," ")#pythonでのtab=\t
    print(lines[i][0:-1])

# 確認コマンド# cat 'popular-names.txt' | sed 's/\t/ /g'