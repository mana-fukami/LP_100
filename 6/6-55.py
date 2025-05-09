#無関係の下位問題を抽出する
#短いコードを書く

# アナロジータスクでの正解率
# 54の実行結果を用い、意味的アナロジー（semantic analogy）と文法的アナロジー（syntactic analogy）の正解率を測定せよ。

# 文法セクションは、[: gram***]と記述されている

from WordVector import Word2Vec

file=open("6/questions-words.txt","r",encoding="utf-8")
lines=file.readlines()
w2v=Word2Vec()
record=open(r"6/full_analogy.txt","w",encoding="utf-8")

for line in lines:
    if ":" in line:
        record.write(line)
    else:
        row=line.replace("\n","").split(" ")
        analogy=w2v.analogy(row[0],row[1],row[2])
        most_sim=analogy[0]
        record.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {most_sim[0]} {most_sim[1]}\n")

file=open(r"6/full_analogy.txt","r",encoding="utf-8")
lines=file.readlines()

flag=None
sem_data=[]
syn_data=[]
for line in lines:
    if ": gram" in line:
        flag="syn"
    elif ":" in line:
        flag="sem"
    elif flag=="sem":
        row=line.replace("\n","").split(" ")
        sem_data.append(row)
    elif flag=="syn":
        row=line.replace("\n","").split(" ")
        syn_data.append(row)

sem_acc=0
for row in sem_data:
    if row[3]==row[4]:
        sem_acc+=1
sem_acc=sem_acc/len(sem_data)

syn_acc=0
for row in syn_data:
    if row[3]==row[4]:
        syn_acc+=1
syn_acc=syn_acc/len(syn_data)

print(f"""意味的アナロジー: {sem_acc}
文法的アナロジー: {syn_acc}""")

# 実行結果
"""
意味的アナロジー: 0.7308602999210734
文法的アナロジー: 0.7414685582822086
"""