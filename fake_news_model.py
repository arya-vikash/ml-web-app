import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle


def tfidf_predict(fake_test):
    model_fake_news = pickle.load(open('fake_news.pkl', 'rb'))
    data=pd.read_csv('https://raw.githubusercontent.com/arya-vikash/csv-files/main/news.csv')
    x_train,x_test,y_train,y_test=train_test_split(data['text'],data['label'],test_size=0.2,random_state=42)
    tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_df=0.7)
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)
    tfidf_fake_test=tfidf_vectorizer.transform([fake_test])
    y=model_fake_news.predict(tfidf_fake_test)
    return y