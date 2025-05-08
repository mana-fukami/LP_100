#無関係の下位問題を抽出する
#短いコードを書く

# 多肢選択問題の正解率
# JMMLU のいずれかの科目を大規模言語モデルに解答させ、その正解率を求めよ。

from API import APIManager
api_manager=APIManager()
api_manager.setting("Gemini")

import csv
import requests
from io import StringIO

# 解答させるデータの読み込み
url="https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/miscellaneous.csv"
jmmlu_file=requests.get(url)
csv_data=StringIO(jmmlu_file.text)
csv_reader=csv.reader(csv_data)

answers=[]
for row in csv_reader:
    question=row[0]
    choice1=row[1]
    choice2=row[2]
    choice3=row[3]
    choice4=row[4]
    answer=row[5]
    prompt=f"""
次の問題の正解を選択肢から1つ選んでください。
問題：{question}
選択肢
A:{choice1}
B:{choice2}
C:{choice3}
D:{choice4}
"""
    instruction="""出力は答えのみ。1文字で答えてください。"""
    response=api_manager.get_response(prompt,instruction)
    answers.append((answer,response.replace("\n","")))

correct=0
all_question=len(answers)
for answer in answers:
    print(f"{answer[0]} : {answer[1]}")
    if answer[0]==answer[1]:
        correct+=1

print(f"Accuracy:{correct/all_question}")

#実行結果
# Accuracy:0.9066666666666666
#以下応答の確認
"""
B : B
D : D
C : A
B : B
C : C
C : C
A : B
A : A
B : B
A : A
B : C
B : B
B : B
A : A
B : B
A : A
A : A
A : A
A : A
D : D
A : A
C : C
B : C
D : D
C : C
C : C
C : C
D : D
A : A
D : D
C : B
C : C
D : D
C : C
C : C
C : C
D : D
B : B
B : B
C : C
A : A
B : B
C : C
D : D
B : B
A : A
A : A
C : C
C : A
B : B
A : A
C : C
C : C
B : B
A : A
C : C
D : D
B : B
B : A
B : B
C : C
C : A
D : D
C : C
A : A
A : A
B : B
A : A
B : B
A : A
C : C
B : B
B : B
C : C
C : C
B : B
C : C
C : C
B : A
C : C
B : B
C : C
D : B
C : C
B : B
C : C
D : D
B : B
C : C
B : B
D : D
B : B
D : D
A : A
D : D
B : B
C : C
B : B
D : D
D : D
C : C
B : B
B : B
C : C
A : A
A : A
C : C
B : B
D : D
D : D
A : A
C : C
A : A
D : D
A : A
C : C
B : B
D : D
B : B
A : A
D : D
B : B
B : B
C : C
C : D
B : B
B : B
D : A
D : D
A : A
C : C
D : D
C : C
C : C
B : D
C : C
C : D
A : A
C : C
C : C
C : C
B : B
B : B
D : D
D : D
B : B
C : C
D : D
D : D
D : D
"""