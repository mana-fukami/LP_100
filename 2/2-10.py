
original_file=open("2\popular-names.txt","r")

#2-10
#行数をカウントする
lines=original_file.readlines()
count_lines=len(lines)
print("行数："+str(count_lines))

#確認
# wc -l popular-names.txt
