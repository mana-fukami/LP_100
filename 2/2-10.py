#美しさ
#コメントすべき事を知る
#コメントは正確で簡潔に

original_file=open("2\popular-names.txt","r")

#2-10
#行数(改行="\n")をカウントする
lines=original_file.readlines()
count_lines=len(lines)
print("行数："+str(count_lines))

# 実行結果
# 行数：2780

# 確認
# wc -l "popular-names.txt"
# 実行結果
# 2780 popular-names.txt
