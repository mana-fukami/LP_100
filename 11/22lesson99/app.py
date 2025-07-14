# app.py
from flask import Flask, request, render_template
from model_def import load_model_and_dict, translate

app = Flask(__name__)

# モデルと辞書の初期化
model, ja_token2id, en_token2id, en_id2token = load_model_and_dict()

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    if request.method == "POST":
        user_input = request.form["input_text"]
        translation = translate(model, ja_token2id, en_token2id, en_id2token, user_input)
    return render_template("index.html", translation=translation)

if __name__ == "__main__":
    app.run(debug=True)
