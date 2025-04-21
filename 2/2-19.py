#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#3列目の数値の逆順でファイルの各行を整列せよ（注意: 各行の内容は変更せずに並び替えよ）。
# 同様の処理をsortコマンドで実現せよ。

original_file=open("popular-names.txt","r")
lines=original_file.readlines()

#後で整列されるために、3列目の数値だけでなく、何行目かも記録しておく。
row3={} #{key:何行目,value:各行3列目の数値(文字列)}
for i in range(len(lines)):
    words=lines[i].replace("\t"," ").split(" ")
    row3[i]=words[2]

#3列目の数値の逆順で整列させる。
#注意：3列目の数値は文字列として扱われている
row3_sorted=sorted(row3.items(),key=lambda x:int(x[1]),reverse=True) #3列目の文字列を数値intとして扱う

#整列して後の情報を書き込むための新しいファイル
sorted_file=open("sorted-popular-names.txt","w")

for i in range(len(row3_sorted)):
    sorted_file.write(lines[row3_sorted[i][0]])

# 実行結果

#確認コマンド#cat 'popular-names.txt' | sed 's/\t/ /g' |sort -k 3 -t " " -r -n
