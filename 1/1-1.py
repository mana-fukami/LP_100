#名前に情報を埋め込む
#誤解されない名前をつける

#「パタトクカシーー」という文字列の2, 4, 6, 8文字目を取り出して連結した文字列を得よ．

original_text="パタトクカシーー"
picked_text=""

for i in range(1,len(original_text)+1,2):
    picked_text+=original_text[i]
print(picked_text)