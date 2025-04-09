#ファイルの先頭10行に対して、各行の1列目だけを抜き出して表示せよ。
# 確認にはcutコマンドなどを用いよ。

original_file=open("popular-names.txt")
lines=original_file.readlines()

for i in range(10):
    lines[i]=lines[i].replace("\t"," ")
    row=lines[i].split(" ")
    print(row[0])

#確認コマンド#  cat 'popular-names.txt' | sed 's/\t/ /g' |cut -f 1 -d " "