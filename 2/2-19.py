#3列目の数値の逆順でファイルの各行を整列せよ（注意: 各行の内容は変更せずに並び替えよ）。
# 同様の処理をsortコマンドで実現せよ。

original_file=open("popular-names.txt","r")
lines=original_file.readlines()
row3={} #キー：何行目、値：行の3列目の値←※文字列
for i in range(len(lines)):
    words=lines[i].replace("\t"," ").split(" ")
    row3[i]=words[2]

row3_sorted=sorted(row3.items(),key=lambda x:int(x[1]),reverse=True) #3列目の文字列を数値intとして扱う

sorted_file=open("sorted-popular-names.txt","w")

for i in range(len(row3_sorted)):
    sorted_file.write(lines[row3_sorted[i][0]])

#確認コマンド#cat 'popular-names.txt' | sed 's/\t/ /g' |sort -k 3 -t " " -r -n
