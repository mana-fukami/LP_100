#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

#ファイルの末尾N行だけを表示せよ。例えば、N=10として末尾10行を表示せよ。
# 確認にはtailコマンドを用いよ。

original_file=open("2\popular-names.txt","r")

lines=original_file.readlines()

N=10
for i in range(N):
    print(lines[-N+i][0:-1]) #N=10のとき、-10,-9,-8,…,-1と表示させたい。

# 実行結果
"""
Liam    M       19837   2018
Noah    M       18267   2018
William M       14516   2018
James   M       13525   2018
Oliver  M       13389   2018
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018
"""

# 確認コマンド# tail -10 'popular-names.txt'
# 実行結果
"""
Liam    M       19837   2018
Noah    M       18267   2018
William M       14516   2018
James   M       13525   2018
Oliver  M       13389   2018
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018
"""