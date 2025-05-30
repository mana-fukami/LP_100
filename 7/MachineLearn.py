import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix

class DataSummary:
    def __init__(self,df):
        self.df=df

    # ポジティブ事例とネガティブ事例を数える
    def count_pos_neg(self):
        pos=0
        neg=0
        for i in range(self.df.shape[0]):
            if self.df.loc[i,"label"]==1:
                pos+=1
            else:
                neg+=1
        return pos,neg

    # 特徴ベクトルに変換する
    def feature_dict(self,sentence):
        feature={}
        splitted=sentence.split(" ")
        for word in splitted:
            if word!="":
                if word not in feature:
                    feature[word]=1
                else:
                    feature[word]+=1
        return feature

    # テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトにまとめる
    def organize_data(self):
        dict_list=[]
        df=self.df
        for i in range(df.shape[0]):
            organized={}
            organized["text"]=df.loc[i,"sentence"]
            organized["label"]=df.loc[i,"label"]
            organized["feature"]=self.feature_dict(df.loc[i,"sentence"])
            dict_list.append(organized)
        return dict_list

class LogisticLearning():
    def __init__(self,df):
        self.df=df
        self.vec=DictVectorizer(sparse=False)
        self.model=self.learned_model()

    def learned_model(self):
        # テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
        train_dict_list=DataSummary(self.df).organize_data()
        # データフレーム型に変換
        train_data=pd.DataFrame(train_dict_list)
        # 学習用の入力値と目標値
        x=self.vec.fit_transform(train_data["feature"])
        t=train_data["label"]
        # 学習
        log_model=LogisticRegression()
        log_model.fit(x,t)
        return log_model

    def predict_data(self,pred_df,n):
        # テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
        dev_dict_list=DataSummary(pred_df).organize_data()
        # データフレーム型に変換
        train_data=pd.DataFrame(dev_dict_list)
        # 予測用の入力値と目標値
        x_pred=self.vec.transform(train_data["feature"])
        t_pred=train_data["label"]
        model_t=self.model.predict(x_pred[:n])
        data_t=t_pred[:n]
        return model_t,data_t

    def predict_text(self,sentence):
        feature={}
        splitted=sentence.split(" ")
        for word in splitted:
            if word!="":
                if word not in feature:
                    feature[word]=1
                else:
                    feature[word]+=1
        x_pred=self.vec.transform(feature)
        predict=self.model.predict(x_pred)
        return predict

    def probs_data(self,pred_df,n):
        # テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
        dev_dict_list=DataSummary(pred_df).organize_data()
        # データフレーム型に変換
        train_data=pd.DataFrame(dev_dict_list)
        # 予測用の入力値と目標値
        x_pred=self.vec.transform(train_data["feature"])
        x_sample=x_pred[:n]
        # 条件付き確率を求める
        probs=self.model.predict_proba([x_sample])
        return probs

    def confusion(self,pred_df):
        # テキスト、特徴ベクトル、ラベルを格納した辞書オブジェクトの作成
        dev_dict_list=DataSummary(pred_df).organize_data()
        # データフレーム型に変換
        train_data=pd.DataFrame(dev_dict_list)
        # 予測用の入力値と目標値
        x_pred=self.vec.transform(train_data["feature"])
        t_pred=train_data["label"]
        # 予測
        y_pred=self.model.predict(x_pred)
        # 混同行列を求める
        cm=confusion_matrix(t_pred,y_pred,labels=self.model.classes_)
        return cm
