#名前に情報を埋め込む
#誤解されない名前をつける

#“Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.”という文を単語に分解し，
# 1, 5, 6, 7, 8, 9, 15, 16, 19番目の単語は先頭の1文字，それ以外の単語は先頭の2文字を取り出し，
# 取り出した文字列から単語の位置（先頭から何番目の単語か）への連想配列（辞書型もしくはマップ型）を作成せよ．

original_text="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words_list=original_text.replace(",","").replace(".","").split(" ")

id_of_pickup_first_char=[1, 5, 6, 7, 8, 9, 15, 16, 19]
recall_dict={}
for i,word in enumerate(words_list):
    if i+1 in id_of_pickup_first_char:
        recall_dict[word[0]]=i+1
    else:
        recall_dict[word[0:2]]=i+1
print(recall_dict)