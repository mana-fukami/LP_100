#1列目の文字列の出現頻度を求め、出現頻度と名前を出現頻度の多い順に並べて表示せよ。
# 確認にはcut, uniq, sortコマンドを用いよ。

original_file=open("popular-names.txt","r")
lines=original_file.readlines()
rows=[]
for i in range(len(lines)):
    lines[i]=lines[i].replace("\t"," ")
    words=lines[i].split(" ")
    rows.append(words[0])

row1_kinds={}
for j in rows:
    if j in row1_kinds:
        row1_kinds[j]+=1
    else:
        row1_kinds[j]=1

row1_sort_by_count=sorted(row1_kinds.items(),key=lambda x:x[1],reverse=True)

for k in row1_sort_by_count:
    print(k[0])

#確認コマンド#cat 'popular-names.txt' | sed 's/\t/ /g' |cut -f 1 -d " "|sort|uniq -c|sort -n -r