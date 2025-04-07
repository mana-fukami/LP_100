#名前に情報を埋め込む
#誤解されない名前をつける

#引数x, y, zを受け取り「x時のyはz」という文字列を返す関数を実装せよ．さらに，x=12, y=”気温”, z=22.4として，実行結果を確認せよ．

def create_message(x, y, z):
    message=str(x)+"時の"+str(y)+"は"+str(z)
    return message

message=create_message(12, "気温", 22.4)
print(message)