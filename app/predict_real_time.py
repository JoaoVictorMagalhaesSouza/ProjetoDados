
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

    # Data Acquisition
    df = DataAcquisition().get_data()
    # %% Data Preparation
    df = DataPreparation(df).normalize_data()

    # Feature Engineering
    df_fe = FeatureEngineering(df).pipeline_feat_eng()
    df['Close_Yesterday'] = df['Close'].shift(1)
    df = df.dropna()
    # Merge dataframes
    df = df.merge(df_fe, left_index=True, right_index=True)
    df_test = df.tail(1)
    #Shuffle the data
    shuffled = df.sample(frac=1)
    df = shuffled.copy()
    

    #Get the last line
    
    df_train = df.iloc[:-1]
    X_train, y_train = df_train.drop('Close', axis = 1), df_train.loc[:,['Close']]
    X_test = df_test.drop('Close', axis = 1)

    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    y_train_xgb = y_train.copy()
    

    

    
    model = ModelXGboost(X_train_xgb, y_train_xgb)
    model.fit()
    result = model.predict(X_test_xgb)[0]
    #import pickle
    #xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    #result = xgb_model_loaded.predict(X_test_xgb)[0]
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    #Increment result in csv
    with open('metricas_diarias.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([today, result])
    
    
