#名前に情報を埋め込む
#誤解されない名前をつける

#スペースで区切られた単語列に対して，各単語の先頭と末尾の文字は残し，それ以外の文字の順序をランダムに並び替えるプログラムを作成せよ．
# ただし，長さが４以下の単語は並び替えないこととする．
# 適当な英語の文
# I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .
# を与え，その実行結果を確認せよ．

import random

def shuffle_words(word):
    if len(word)<=4:
        return word
    else:
        shuffled_word=""
        shuffled_word+=word[0]
        shuffled_word+="".join(random.sample(word[1:-1],len(word[1:-1])))
        shuffled_word+=word[-1]
        return shuffled_word

original_text="I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

words_list=original_text.split(" ")
shuffled_text=""
for i in words_list:
    shuffled_text+=shuffle_words(i)+" "

print(shuffled_text)