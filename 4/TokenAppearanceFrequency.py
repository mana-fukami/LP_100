import MeCab

def token_class_appearance_frequency(frequency_dict,text,token_class):
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(text)
    while node:
        if node.surface != "":
            node_feature=node.feature.split(",")
            if node_feature[0] == token_class:
                if node.surface not in frequency_dict:
                    frequency_dict[node.surface]=1
                else:
                    frequency_dict[node.surface]+=1
        node=node.next
    return frequency_dict

def token_appearance_frequency(frequency_dict,text):
    tagger=MeCab.Tagger(r"C:\Users\mana\AppData\Local\Programs\Python\Python313\Lib\site-packages\unidic\dicdir")
    node=tagger.parseToNode(text)
    while node:
        if node.surface != "":
            if node.surface not in frequency_dict:
                frequency_dict[node.surface]=1
            else:
                frequency_dict[node.surface]+=1
        node=node.next
    return frequency_dict