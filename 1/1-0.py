#名前に情報を埋め込む
#誤解されない名前をつける

#「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

original_text1="パトカー"
original_text2="タクシー"
interleaved_text="" #interleave:動作を交互に行う
length_of_interleaved_text=len(original_text1)+len( original_text2)

id_of_text1=0
id_of_text2=0
for i in range(0,length_of_interleaved_text,1):
    if i%2==0:
        interleaved_text+=original_text1[id_of_text1]
        id_of_text1+=1
    else:
        interleaved_text+= original_text2[id_of_text2]
        id_of_text2+=1
print(interleaved_text)