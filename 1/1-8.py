#名前に情報を埋め込む
#誤解されない名前をつける

#与えられた文字列の各文字を，以下の仕様で変換する関数cipherを実装せよ．
# ・英小文字ならば(219 - 文字コード)の文字に置換
# ・その他の文字はそのまま出力
# この関数を用い，英語のメッセージを暗号化・復号化せよ．

def cipher(text):
    cipher_text=""
    for char in text:
        if char.islower():
            cipher_text+=chr(219-ord(char))
        else:
            cipher_text+=char
    return cipher_text

text="quick"
cipher_text=cipher(text)
print(cipher_text)
decipher_text=cipher(cipher_text)
print(decipher_text)