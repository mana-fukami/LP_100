#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#ファイルを行単位でN分割し、別のファイルに格納せよ。
# 例えば、N=10としてファイルを10分割せよ。
# 同様の処理をsplitコマンドで実現せよ。
# 実行前にfolder:2に移動しておく。(cd 2)

original_file=open("popular-names.txt","r")
lines=original_file.readlines()
max_lines=len(lines)

N=10
one_file_lines=max_lines//N #分割後のファイル1つの行数#整数割り算→//,小数割り算→/
for i in range (N):
    new_file=open("popular-names"+str(i)+".txt","w") #分割したファイル名の指定
    for j in range(one_file_lines):
        new_file.write(lines[i*one_file_lines+j]) #分割された(i+1)個目のファイルに書き込む

#確認コマンド# split -n l/10 popular-names.txt