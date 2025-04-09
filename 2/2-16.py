#ファイルを行単位でランダムに並び替えよ
# （注意: 各行の内容は変更せずに並び替えよ）。
# 同様の処理をshufコマンドで実現せよ。

original_file = open("popular-names.txt", "r")
lines= original_file.readlines()

new_file= open("popular-names-random.txt", "w")

import random
new_file.writelines(random.sample(lines, len(lines)))

#確認コマンド# shuf popular-names.txt > popular-names-random.txt