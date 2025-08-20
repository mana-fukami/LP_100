# app.py
from flask import Flask, request, render_template
from model_def import load_model_and_dict, translate

app = Flask(__name__)

# アプリケーション起動時に、モデル、辞書、Taggerを一度だけ初期化します
print("Loading model and dictionaries...")
model, ja_token2id, en_token2id, en_id2token, tagger = load_model_and_dict()
print("Model and dictionaries loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    # 翻訳後も入力した日本語がテキストエリアに残るように、元のテキストを取得します
    original_text = request.form.get("input_text", "")
    
    if request.method == "POST":
        user_input = request.form["input_text"]
        if user_input: # 入力がある場合のみ翻訳を実行
            print(f"Translating: {user_input}")
            # translate関数に初期化済みのtaggerを渡します
            translation = translate(model, ja_token2id, en_token2id, en_id2token, tagger, user_input)
            print(f"Translation result: {translation}")
    
    return render_template("index.html", translation=translation, original_text=original_text)

if __name__ == "__main__":
    app.run(debug=True)
