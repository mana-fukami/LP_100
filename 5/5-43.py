#無関係の下位問題を抽出する
#短いコードを書く

# 応答のバイアス
# 問題42において、実験設定を変化させると正解率が変化するかどうかを調べよ。
# 実験設定の例としては、大規模言語モデルの温度パラメータ、プロンプト、多肢選択肢の順番、
# 多肢選択肢の記号などが考えられる。
# 正解の選択肢を全てDに入れ替えて解答させる例。

import requests
from io import StringIO
from Difference import MakeDiff

# 解答させるデータの読み込み
url="https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/miscellaneous.csv"
jmmlu_file=requests.get(url)
csv_data=StringIO(jmmlu_file.text)

make_diff=MakeDiff(csv_data)

def acc(answers):
    correct=0
    all_question=len(answers)
    for answer in answers:
        if answer[0]==answer[1]:
            correct+=1
    return correct/all_question


diff_temp1=make_diff.diff_temp(0.5)
diff_temp1_acc=acc(diff_temp1)
print(f"[temp=0.5]Acc={diff_temp1_acc}")
# 実行例
# [temp=0.5]Acc=0.9133333333333333

#diff_prompt=make_diff.diff_prompt()
#diff_prompt_acc=acc(diff_prompt)
#print(f"[diff-prompt]Acc={diff_prompt_acc}")
# 実行例
# [diff-prompt]Acc=0.9133333333333333

#diff_choice_ord=make_diff.all_answer_d()
#diff_choice_ord_acc=acc(diff_choice_ord)
#print(f"[diff-ord].Acc={diff_choice_ord_acc}")
# 実行例
# [diff-ord].Acc=0.20666666666666667