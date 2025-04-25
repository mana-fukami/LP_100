import re

#textに含まれるWikipwdiaマークアップの除去
def remove_markup(text):
    #強調マークアップの除去
    text=re.sub("'*?","",text)
    #内部リンクの除去,表示文字は残す
    text=re.sub("\[\[([^\[]+)\|([^\[]+)\|","",text)
    text=re.sub("\[\[([^\[]+)\|","",text)
    text=re.sub("\[\[","",text)
    text=re.sub("\]\]","",text)
    #ファイルのマークアップの除去,説明文は残す
    text=re.sub("ファイル.*?\|.*?\|","",text)
    #外部リンクの除去
    text=re.sub("\[http.*?\]","",text)
    #カテゴリの除去
    text=re.sub("Category","",text)
    #リダイレクトの除去,記事名と説明は残す
    text=re.sub("\#REDIRECT","",text)
    #Cite関連の除去
    text=re.sub("\{\{Cite web .+\}\}","",text)
    text=re.sub("\{\{Cite journal .+\}\}","",text)
    text=re.sub("\{\{Cite book .+\}\}","",text)
    #テンプレートの除去
    text=re.sub("\{\{.*?\|.*?\|","",text)
    text=re.sub("\{\{.*?\|","",text)
    text=re.sub("\{\{","",text)
    text=re.sub("\}\}","",text)
    #<ref>~</ref>の除去
    text=re.sub("\<ref.*?\>.*</ref>","",text)
    #コメントアウトなど<~>の除去
    text=re.sub("\<.*?\>","",text)
    #そのほか記号の除去
    text=re.sub("=*?","",text)
    text=re.sub("\|","",text)
    text=re.sub("\;|\:","",text)
    text=re.sub("\*|\#","",text)
    return text