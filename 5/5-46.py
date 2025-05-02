# 川柳の生成
# 適当なお題を設定し、川柳の案を10個作成せよ。

from API import APIManager

api_manager=APIManager()
api_manager.setting("Gemini")

question="""次のお題に関する川柳の案を10個作成してください。
お題「溶けかけのアイス」"""

response=api_manager.get_response(question)

print(response,end="")

#実行例
"""
## 溶けかけのアイス 川柳案 10選

1.  指先に  甘い雫の  夏の罪
2.  時間よ止まれ  必死の形相で  頬張る夏
3.  幸せも  溶ける速度も  加速する
4.  溶けてゆく  夢の残骸  ハンカチへ
5.  あわててさ  吸い込む先に  脳天パンチ
6.  溶けかけた  アイスに似てる  恋心
7.  ベタベタと  後悔残る  夏の午後
8.  滴るは  ミルクの涙か  夏の汗
9.  「もう一口」  迷う指先  溶けるアイス
10. 溶けかけの  アイス見つめて  哲学す
"""