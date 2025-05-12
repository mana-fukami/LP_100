from gensim.models import KeyedVectors
import numpy as np

class Word2Vec:
    def __init__(self):
        self.model_path="6\GoogleNews-vectors-negative300.bin"
        self.model=KeyedVectors.load_word2vec_format(self.model_path,binary=True)

    def get_vector(self,word):
        word=word.replace(" ","_")
        if word in self.model:
            return self.model[word]
        else:
            return None

    def word_cos_sim(self,w1,w2):
        """
        v1=self.get_vector(w1)
        v2=self.get_vector(w2)
        dot_product=np.dot(v1,v2)
        norm1=np.linalg.norm(v1)
        norm2=np.linalg.norm(v2)
        return dot_product/(norm1*norm2)
        """
        w1=w1.replace(" ","_")
        w2=w2.replace(" ","_")
        return self.model.similarity(w1,w2)

    def word_cos_sim_search(self,search_word):
        """
        sim_list=[]
        for word in self.model.key_to_index:
            sim=self.word_cos_sim(search_word,word)
            sim_list.append((word,sim))
        sim_list.sort(key=lambda x:x[1],reverse=True)
        return sim_list
        """
        search_word=search_word.replace(" ","_")
        return self.model.most_similar(search_word)

    def vec_cos_sim_search(self,search_vec):
        sim_list=[]
        for word in self.model.key_to_index:
            vec=self.get_vector(word)
            sim=self.vec_cos_sim(search_vec,vec)
            sim_list.append((word,sim))
        sim_list.sort(key=lambda x:x[1],reverse=True)
        return sim_list

    def vec_sub_add(self,w1,w2,w3):
        v1=self.get_vector(w1)
        v2=self.get_vector(w2)
        v3=self.get_vector(w3)
        result=v1-v2+v3
        return result

    def analogy(self,w1,w2,w3):
        w1=w1.replace(" ","_")
        w2=w2.replace(" ","_")
        w3=w3.replace(" ","_")
        return self.model.most_similar(positive=[w2,w3],negative=[w1])