import os
import pandas as pd
import numpy as np
import tensorflow as tf
from zipfile import ZipFile

model=tf.keras.models.load_model('./models/kostenko_model_1.h5')

def predict_df(df_list):
    idx=1
    for df in df_list:
        X=df.iloc[:, 2:].astype('float')
        print(X)
        print(X.info())
        y_hat=model.predict(X)
        print(type(y_hat))
        y_hat=list(map(lambda x: 'Churn' if x>=0.5 else 'No churn' if x<0.5 else '-----', y_hat))
        df['Prediction']=y_hat
        df.to_csv(f'./files/csv/prediction_{idx}.csv')
        idx+=1


def archive():
    zipObj=ZipFile('./files/prediction.zip', 'w')
    path='./files/csv'
    files=os.listdir(path='./files/csv')
    for file in files:
        zipObj.write(os.path.join(path, file))
    zipObj.close()

