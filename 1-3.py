#名前に情報を埋め込む
#誤解されない名前をつける

#“Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.”という文を単語に分解し，各単語の（アルファベットの）文字数を先頭から出現順に並べたリストを作成せよ．

original_text="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
words_list=original_text.replace(",","").replace(".","").split(" ")
word_length_list=[]

for word in words_list:
    word_length_list.append(len(word))
print(word_length_list)
