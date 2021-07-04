import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_data=pd.read_csv('wine_pre_scaled_data.csv')
scaler=MinMaxScaler()
scaler.fit(train_data)

def data_preprocessor(df):
    if df['type'][0]=='white':
        df['type_white']=int(1)
    else:
        df['type_white']=int(0)
    df.drop('type',axis=1,inplace=True)
    df_scaled=scaler.transform(df)
    return df_scaled
def predict_wine_quality(data,model):
    if type(data)==dict:
        df=pd.DataFrame(data)
    else:
        df=data
    scaled_data=data_preprocessor(df)
    y_pred=model.predict(scaled_data)
    return y_pred