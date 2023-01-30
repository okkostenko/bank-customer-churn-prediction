import os
import pandas as pd
import numpy as np
import tensorflow as tf
from zipfile import ZipFile

df=pd.read_csv('./notebooks/Churn_Modelling.csv')

def numerical_prepocessing(df):
    saved_values={}
    columns2save=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Exited']
    for column in columns2save:
        saved_values[column]=[df[column].min(), df[column].mean(), df[column].max()]
    return saved_values

saved=numerical_prepocessing(df)

def recalculate_numerical(value, feature_name):
    saved_values=saved
    value=(value-saved_values[feature_name][0])/(saved_values[feature_name][2]-saved_values[feature_name][0])
    return value

def encode_categorical(value, feature_name):
    if feature_name=='geography':
        enc=[0, 0, 0]
    else:
        enc=[0, 0]
    enc[value]=1

    return enc

def process_df(df_input):
    df=df_input.copy()
    del df['Exited']
    numerical_features=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    categorical_features=['Geography', 'Gender']
    for feature in numerical_features:
        df[feature]=list(map(lambda x: (x-saved[feature][0])/(saved[feature][2]-saved[feature][0]), df[feature]))
    df=pd.get_dummies(df, columns=categorical_features)
    #df[['Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male']].astype('str').astype('int')
    df=df.loc[:, ['CustomerId', 'Surname', 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male']]
    
    return df


   


