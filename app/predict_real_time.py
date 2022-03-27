
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

from utils.models import  ModelXGboost


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
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    #Increment result in csv
    with open('metricas_diarias.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([today, result])
    
    
