#名前に情報を埋め込む
#誤解されない名前をつける

#文字列”stressed”の文字を逆に（末尾から先頭に向かって）並べた文字列を得よ．

original_text="stressed"
reversed_text=""

for i in range(len(original_text)-1,-1,-1):
    reversed_text+=original_text[i]
print(reversed_text)
