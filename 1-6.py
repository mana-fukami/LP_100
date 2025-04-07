#名前に情報を埋め込む
#誤解されない名前をつける

#“paraparaparadise”と”paragraph”に含まれる文字bi-gramの集合を，それぞれ, XとYとして求め，
# XとYの和集合，積集合，差集合を求めよ．さらに，’se’というbi-gramがXおよびYに含まれるかどうかを調べよ．

def n_gram(sequence,n):
    n_gram_list=[]
    for i in range(len(sequence)-n+1):
        n_gram_list.append(sequence[i:i+n])
    return n_gram_list

original_text1="paraparaparadise"
original_text2="paragraph"
X=n_gram(original_text1.replace(" ",""),2)
Y=n_gram(original_text2.replace(" ",""),2)

X_set=set(X)
Y_set=set(Y)
print("X_set:",X_set)
print("Y_set:",Y_set)
#和集合
union_set=X_set.union(Y_set)
print("union_set:",union_set)
#積集合
interaction_set=X_set.intersection(Y_set)
print("interaction_set:",interaction_set)
#差集合
difference_set=X_set.difference(Y_set)
print("difference_set:",difference_set)

if "se" in X_set:
    print("seはXに含まれます")
else:
    print("seはXに含まれません")

if "se" in Y_set:
    print("seはYに含まれます")
else:
    print("seはYに含まれません")