# Imports
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
from utils.correlation_analysis import CorrelationAnalysis
from utils.split import split_train_test
from utils.models import ModelLSTM, ModelXGboost, ModelCatboost

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
    # Correlation Analysis -  A fazer :

    #Get the last line
    df_test = df.tail(1)
    df_train = df.iloc[:-1]
    X_train, y_train = df_train.drop('Close', axis = 1), df_train.loc[:,['Close']]
    X_test = df_test.drop('Close', axis = 1)

    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    y_train_xgb = y_train.copy()
    

    

    
    model = ModelXGboost(X_train_xgb, X_test_xgb, y_train_xgb)
    model.fit()
    result = model.predict()[0]
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    #Increment result in csv
    with open('metricas_diarias.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([today, result])
    
    
