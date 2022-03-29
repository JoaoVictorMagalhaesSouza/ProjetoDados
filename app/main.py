
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

#https://towardsdatascience.com/predictive-analysis-rnn-lstm-and-gru-to-predict-water-consumption-e6bb3c2b4b02
#Defining the working directory for internal functions
import os, sys
dir_import = os.getcwd()
sys.path.insert(0,dir_import+'/../')
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
from utils.models import  ModelXGboost

def realizar_predicao(nome_modelo: str):

    # Data Acquisition
    df = DataAcquisition().get_data()
    # %% Data Preparation
    df = DataPreparation(df).normalize_data()

    # Feature Engineering
    df_fe = FeatureEngineering(df).pipeline_feat_eng()
    df['Close_Yesterday'] = df['Close'].shift(1)
    df['Close_Tomorrow'] = df['Close'].shift(-1)
    df = df.drop('Adj Close', axis = 1)
    
    # Merge dataframes
    df = df.merge(df_fe, left_index=True, right_index=True)
   

    
    df = df.dropna()
    df_test = df.iloc[(int(len(df)*0.9)):]
    X_test, y_test = df_test.drop('Close_Tomorrow', axis = 1), df_test.loc[:,['Close_Tomorrow']]
    #Shuffle the data
    df_train_val = df.iloc[:(int(len(df)*0.9))]
    df_shuffled = df_train_val.sample(frac=1)
    df_train_shuffled = df_shuffled
    X_train, y_train = df_train_shuffled.drop('Close_Tomorrow', axis = 1), df_train_shuffled.loc[:,['Close_Tomorrow']]
    
    index_test = y_test.index
    # Scaller

    
    if (nome_modelo == 'XGBoost'):
        model = ModelXGboost(X_train, y_train)
        model.fit()
        result = model.predict(X_test)
        real = y_test.values.copy()
        dict_output = {'y_true': real.flatten(),'y_pred': result, 'index': index_test}
        print(pd.DataFrame(dict_output))
        return pd.DataFrame(dict_output)


def calcula_metrica(y_true, y_pred, index):
    mae = mean_absolute_error(y_pred, y_true)
    mse = mean_squared_error(y_pred, y_true)
    percentual_dif = 0
    for r,p in zip(y_pred,y_true):
         percentual_dif += (abs(r-p)/r)
        
    return mae, mse, percentual_dif


def make_fig(dataframe):
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    y_true,y_pred,index = dataframe.y_true, dataframe.y_pred, dataframe['index']
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