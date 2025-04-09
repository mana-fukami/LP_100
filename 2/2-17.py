#1列目の文字列の異なり（文字列の種類）を求めよ。
# 確認にはcut, sort, uniqコマンドを用いよ。

original_file=open("popular-names.txt","r")
lines=original_file.readlines()
rows=[]
for i in range(len(lines)):
    lines[i]=lines[i].replace("\t"," ")
    words=lines[i].split(" ")
    rows.append(words[0])

row1_kinds=[]
for j in rows:
    if j not in row1_kinds:
        row1_kinds.append(j)
row1_kinds.sort()

for k in row1_kinds:
    print(k)

#確認コマンド# cat 'popular-names.txt' | sed 's/\t/ /g' |cut -f 1 -d " "|sort|uniq