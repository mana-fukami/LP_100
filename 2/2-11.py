#ファイルの先頭N行だけを表示せよ。例えば、N=10として先頭10行を表示せよ。確認にはheadコマンドを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()

N=10
for i in range(N):
    print(lines[i][1:-1])#lines[i]にも改行は含まれる。＋print()でも末尾に改行がある→lines[i][1:-1]で改行を削除して表示。