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
   

    df = df.dropna()
    df_test = df.iloc[(int(len(df)*0.9)):]
    X_test, y_test = df_test.drop('Close', axis = 1), df_test.loc[:,['Close']]
    #Shuffle the data
    df_train_val = df.iloc[:(int(len(df)*0.9))]
    df_shuffled = df_train_val.sample(frac=1)
    df_train_shuffled = df_shuffled.iloc[:(int(len(df_shuffled)*0.8))]
    df_val_shuffled = df_shuffled.iloc[(int(len(df_shuffled)*0.8)):]
    X_train, y_train = df_train_shuffled.drop('Close', axis = 1), df_train_shuffled.loc[:,['Close']]
    X_val, y_val = df_val_shuffled.drop('Close', axis = 1), df_val_shuffled.loc[:,['Close']]
    index_val = y_val.index
    index_test = y_test.index
    # Scaller

    
    if (nome_modelo == 'XGBoost'):
        model = ModelXGboost(X_train, y_train)
        model.fit()
        result = model.predict(X_test)
        real = y_test.values.copy()
        dict_output = {'y_true': real.flatten(),'y_pred': result, 'index': index_test}
        return dict_output


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
    mode='markers+lines',
    marker_color='#000000',
    ), secondary_y=False)

    
    fig.add_trace(
        go.Scatter(
        x=index,
        y=y_pred,
        name='BTC Previsto',
        mode='markers+lines',
        marker_color='#fd5800',#'#ccff33',
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
            height=350,
            title={'text': 'Previsão do BTC - Conjunto de Testes', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
            )
    return fig

'''
    Criar a aplicação em tempo real
    Terminar de ajustar a tela
    Subir a tela.
'''