#%% Imports
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
#https://towardsdatascience.com/predictive-analysis-rnn-lstm-and-gru-to-predict-water-consumption-e6bb3c2b4b02
#Defining the working directory for internal functions
import os, sys
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
from utils.models import ModelXGboost
import numpy as np
#Import standard scaller
from plotly.subplots import make_subplots

#%% Data Acquisition
df = DataAcquisition().get_data()
# %% Data Preparation
df = DataPreparation(df).normalize_data()

#%% Feature Engineering
df_fe = FeatureEngineering(df).pipeline_feat_eng()
df['Close_Yesterday'] = df['Close'].shift(1)
df['Close_Tomorrow'] = df['Close'].shift(-1)
df = df.drop('Adj Close', axis = 1)
#df = df.dropna()

#%% Merge dataframes
df = df.merge(df_fe, left_index=True, right_index=True)

# %% Split data in 80 %train, 10% val and 10% test
df = df.dropna()
df_test = df.iloc[(int(len(df)*0.9)):]
X_test, y_test = df_test.drop('Close_Tomorrow', axis = 1), df_test.loc[:,['Close_Tomorrow']]
#Shuffle the data
df_train_val = df.iloc[:(int(len(df)*0.9))]
df_shuffled = df_train_val.sample(frac=1)
df_train_shuffled = df_shuffled.iloc[:(int(len(df_shuffled)*0.8))]
df_val_shuffled = df_shuffled.iloc[(int(len(df_shuffled)*0.8)):]
X_train, y_train = df_train_shuffled.drop('Close_Tomorrow', axis = 1), df_train_shuffled.loc[:,['Close_Tomorrow']]
X_val, y_val = df_val_shuffled.drop('Close_Tomorrow', axis = 1), df_val_shuffled.loc[:,['Close_Tomorrow']]
index_val = y_val.index
index_test = y_test.index

#%%Plots
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
    fig.show()
    
#%% Model XGBoost
model = ModelXGboost(X_train, y_train)
model.fit()
#%%Validation
result_validation = model.predict(X_val)
predicao = result_validation.copy()
real = y_val.values.copy()
mae = mean_absolute_error(predicao, real)
mse = mean_squared_error(predicao, real)
print('MAE de validação: ', mae)
print('MSE de validação: ', mse)
#Percentual de erro
percentual_dif = 0
for r,p in zip(predicao,real):
    percentual_dif += (abs(r-p)/r)
print('Percentual de erro do XGboost na validação: +-', round(percentual_dif[0],2),"%")
make_fig(real.flatten(),predicao,index_val,'XGboost')

#%%Test
result_test = model.predict(X_test)
predicao = result_test.copy()
real = y_test.values.copy()
mae = mean_absolute_error(predicao, real)
mse = mean_squared_error(predicao, real)
print('MAE de teste: ', mae)
print('MSE de teste: ', mse)
#Percentual de erro
percentual_dif = 0
for r,p in zip(predicao,real):
    percentual_dif += (abs(r-p)/r)
print('Percentual de erro do XGboost no teste: +-', round(percentual_dif[0],2),"%")
make_fig(real.flatten(),predicao,index_test,'XGboost')

#%% Save model
import pickle
file_name = "xgb_reg.pkl"

# save
pickle.dump(model, open(file_name, "wb"))

# load
xgb_model_loaded = pickle.load(open(file_name, "rb"))

# %%
