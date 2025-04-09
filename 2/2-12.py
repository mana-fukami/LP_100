#ファイルの末尾N行だけを表示せよ。例えば、N=10として末尾10行を表示せよ。
# 確認にはtailコマンドを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()

N=10
for i in range(N):
    print(lines[-N+i][0:-1])#N=10のとき、-10,-9,-8,…と表示させたい。