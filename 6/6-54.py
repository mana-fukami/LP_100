#無関係の下位問題を抽出する
#短いコードを書く

# アナロジーデータでの実験
# 単語アナロジーの評価データをダウンロードし、国と首都に関する事例（: capital-common-countriesセクション）に対して、
# vec(2列目の単語) - vec(1列目の単語) + vec(3列目の単語)を計算し、そのベクトルと類似度が最も高い単語と、その類似度を求めよ。
# 求めた単語と類似度は、各事例と一緒に記録せよ。

from WordVector import Word2Vec
from datetime import datetime

file=open("6/questions-words.txt","r",encoding="utf-8")
lines=file.readlines()

read_flag=False
search_case=": capital-common-countries\n"
data=[]
for line in lines:
    if ":" in line:
        if line==search_case:
            read_flag=True
        else:
            read_flag=False
    elif read_flag==True:
        row=line.replace("\n","").split(" ")
        data.append(row)

w2v=Word2Vec()
now=datetime.now()
record=open(f"6/analogy_result_{now.month}-{now.day}-{now.hour}-{now.minute}.txt","w",encoding="utf-8")
record.write(f"{search_case}")

for row in data:
    #result=w2v.vec_sub_add(row[1],row[0],row[2])
    #sim_list=w2v.vec_cos_sim_search(result)
    sim_list=w2v.analogy(row[0],row[1],row[2])
    most_sim=sim_list[0]
    record.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {most_sim[0]} {most_sim[1]}\n")

# 実行例はanalogy_result_*.textに