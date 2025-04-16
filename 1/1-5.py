#名前に情報を埋め込む
#誤解されない名前をつける

#与えられたシーケンス（文字列やリストなど）からn-gramを作る関数を作成せよ．
# この関数を用い，”I am an NLPer”という文から単語bi-gram，文字tri-gramを得よ．

def n_gram(sequence,n):
    n_gram_list=[]
    for i in range(len(sequence)-n+1):
        n_gram_list.append(sequence[i:i+n])
    return n_gram_list

original_text="I am an NLPer"
word_n_gram=n_gram(original_text.split(" "),2)
char_n_gram=n_gram(original_text.replace(" ",""),3)
print(word_n_gram)
print(char_n_gram)
