import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer



def predict_spam(text):
    model_spam_email=pickle.load(open('spam_email.pkl','rb'))
    data=pd.read_csv('spam_ham_dataset.csv')
    vectorizer=CountVectorizer()
    x=vectorizer.fit_transform(data['text'])
    vect_data=vectorizer.transform([text])
    y=model_spam_email.predict(vect_data)
    return y