#ファイルを行単位でN分割し、別のファイルに格納せよ。
# 例えば、N=10としてファイルを10分割せよ。
# 同様の処理をsplitコマンドで実現せよ。

original_file=open("popular-names.txt","r")
lines=original_file.readlines()
max_lines=len(lines)

N=10
one_file_lines=max_lines//N #整数割り算→//,小数割り算→/
for i in range (N):
    new_file=open("popular-names"+str(i)+".txt","w")
    for j in range(one_file_lines):
        new_file.write(lines[i*one_file_lines+j])

#確認コマンド# split -n l/10 popular-names.txt
#このコマンドだとファイルが均等に10分割されない