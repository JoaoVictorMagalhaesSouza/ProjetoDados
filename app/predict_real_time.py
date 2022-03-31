
import csv
import datetime

#https://towardsdatascience.com/predictive-analysis-rnn-lstm-and-gru-to-predict-water-consumption-e6bb3c2b4b02
#Defining the working directory for internal functions
import os, sys
dir_import = os.getcwd()
sys.path.insert(0,dir_import+'/../')
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
import pandas as pd
from utils.models import  ModelXGboost
from utils.database import cria_conexão_banco

def real_time_prediction():

    df = DataAcquisition().get_data()
    df = DataPreparation(df).normalize_data()
    df_fe = FeatureEngineering(df).pipeline_feat_eng()
    df['Close_Yesterday'] = df['Close'].shift(1)
    df['Close_Tomorrow'] = df['Close'].shift(-1)
    df = df.drop('Adj Close', axis = 1)
    df = df.merge(df_fe, left_index=True, right_index=True)
    
    df_test = df.tail(1)
    X_test = df_test.drop('Close_Tomorrow', axis = 1)

    df_train_val = df.iloc[:-1]
    df_shuffled = df_train_val.sample(frac=1)
    X_train, y_train = df_shuffled.drop('Close_Tomorrow', axis = 1), df_shuffled.loc[:,['Close_Tomorrow']]
    model = ModelXGboost(X_train, y_train)
    model.fit()
    result = model.predict(X_test)[0]
    #import pickle
    #xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    #result = xgb_model_loaded.predict(X_test_xgb)[0]
    tomorrow = datetime.datetime.today() + datetime.timedelta(days=1)
    tomorrow = tomorrow.strftime("%Y-%m-%d %H:%M:%S")
    connect = cria_conexão_banco()
    preco_real = df['Close'][-1]
    query_update = f"UPDATE dados_tempo_real SET BTCReal = {preco_real} WHERE TS=(SELECT MAX(TS) FROM dados_tempo_real)"
    #Send to databasef
    try:
        cursor = connect.cursor()
        cursor.execute(query_update)
        connect.commit()
    except Exception as e:
        print(e)
        connect.rollback()
    query_insert = f"INSERT INTO dados_tempo_real (TS, BTCReal, BTCPrevisto) VALUES ('{tomorrow}', {'NULL'}, '{result}')"
    try:
        cursor = connect.cursor()
        cursor.execute(query_insert)
        connect.commit()
    except Exception as e:
        print(e)
        connect.rollback()
    
    
