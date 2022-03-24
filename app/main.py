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

def realizar_predicao(nome_modelo: str):

    # Data Acquisition
    df = DataAcquisition().get_data()
    # %% Data Preparation
    df = DataPreparation(df).normalize_data()

    # Feature Engineering
    df_fe = FeatureEngineering(df).pipeline_feat_eng()
    # Merge dataframes
    df = df.merge(df_fe, left_index=True, right_index=True)
    # Correlation Analysis -  A fazer :

    # %% Split data
    df = df.dropna()
    #80%train 20%test
    df_train, df_test = df.iloc[:int(len(df)*0.8)], df.iloc[int(len(df)*0.8):]
    print('Dimension of train data: ',df_train.shape)
    print('Dimension of test data: ', df_test.shape)

    # Split train data to X and y
    X_train = df_train.drop('Close', axis = 1)
    y_train = df_train.loc[:,['Close']]
    # Split test data to X and y
    X_test = df_test.drop('Close', axis = 1)
    y_test = df_test.loc[:,['Close']]

    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    y_train_xgb = y_train.copy()
    y_test_xgb = y_test.copy()

    index = y_test.index
    # Scaller

    # Different scaler for input and output
    scaler_x = MinMaxScaler(feature_range = (0,1))
    scaler_y = MinMaxScaler(feature_range = (0,1))
    # Fit the scaler using available training data
    input_scaler = scaler_x.fit(X_train)
    output_scaler = scaler_y.fit(y_train)
    # Apply the scaler to training data
    train_y_norm = output_scaler.transform(y_train)
    train_x_norm = input_scaler.transform(X_train)
    # Apply the scaler to test data
    test_y_norm = output_scaler.transform(y_test)
    test_x_norm = input_scaler.transform(X_test)
    # %% 3D input

    def create_dataset (X, y, time_steps = 1):
        Xs, ys = [], []
        for i in range(len(X)-time_steps):
            v = X[i:i+time_steps, :]
            Xs.append(v)
            ys.append(y[i+time_steps])
        return np.array(Xs), np.array(ys)
    TIME_STEPS = 30
    X_test, y_test = create_dataset(test_x_norm, test_y_norm,   
                                    TIME_STEPS)
    X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                    TIME_STEPS)
    print('X_train.shape: ', X_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('X_test.shape: ', X_test.shape)
    print('y_test.shape: ', y_test.shape)

    dict_output = {}

    if (nome_modelo == 'LSTM'):
    # Model LSTM
        model = ModelLSTM(X_train, X_test, y_train)
        model.fit()
        result = model.predict()
        predicao = output_scaler.inverse_transform(result.reshape(-1,1))
        real = output_scaler.inverse_transform(y_test.reshape(-1,1))
        dict_output = {'y_true': real.flatten(),'y_pred': predicao.flatten(), 'index': index}
        return dict_output
        # Evaluate
        # predicao = output_scaler.inverse_transform(result.reshape(-1,1))
        # real = output_scaler.inverse_transform(y_test.reshape(-1,1))
        # mae = mean_absolute_error(predicao, real)
        # mse = mean_squared_error(predicao, real)
        # print('MAE: ', mae)
        # print('MSE: ', mse)
        # #Percentual de erro
        # percentual_dif = 0
        # for r,p in zip(predicao,real):
        #     percentual_dif += (abs(r-p)/r)
        # print('Percentual de erro da LSTM: +-', round(percentual_dif[0],2),"%")
        # make_fig(real.flatten(),predicao.flatten(),index,'LSTM')
    # %% Model XGBoost
    elif (nome_modelo == 'XGBoost'):
        model = ModelXGboost(X_train_xgb, X_test_xgb, y_train_xgb)
        model.fit()
        result = model.predict()
        real = y_test_xgb.values.copy()
        dict_output = {'y_true': real.flatten(),'y_pred': result, 'index': index}
        return dict_output


    # mae = mean_absolute_error(predicao, real)
    # mse = mean_squared_error(predicao, real)
    # print('MAE: ', mae)
    # print('MSE: ', mse)
    # #Percentual de erro
    # percentual_dif = 0
    # for r,p in zip(predicao,real):
    #     percentual_dif += (abs(r-p)/r)
    # print('Percentual de erro do XGboost: +-', round(percentual_dif[0],2),"%")
    # make_fig(real.flatten(),predicao,index,'XGboost')

def calcula_metrica(y_true, y_pred, index):
    mae = mean_absolute_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)
    percentual_dif = 0
    for r,p in zip(y_pred,y_true):
         percentual_dif += (abs(r-p)/r)
        
    return mae, mse, percentual_dif





def make_fig(y_true,y_pred,index,model_name):
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
    go.Scatter(
    x=index,
    y=y_true,
    name='BTC Real',
    mode='markers+lines'
    ), secondary_y=False)

    
    fig.add_trace(
        go.Scatter(
        x=index,
        y=y_pred,
        name='BTC Previsto',
        mode='markers+lines'
    ), secondary_y=False)

   

    fig.update_yaxes(
        title_text="Preço",
        
            secondary_y=False, 
            gridcolor='#d3d3d3', 
            zerolinecolor='black')

    fig.update_xaxes(
        title_text="Data",
            gridcolor='#d3d3d3', 
            zerolinecolor='black')

    fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=100, r=0, b=50, t=50),
            height=350
            )
    return fig