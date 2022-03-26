#%% Imports
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
import optuna
import xgboost as xgb
#Import standard scaller
from sklearn.preprocessing import StandardScaler

#%% Data Acquisition
df = DataAcquisition().get_data()
# %% Data Preparation
df = DataPreparation(df).normalize_data()

#%% Feature Engineering
df_fe = FeatureEngineering(df).pipeline_feat_eng()
df['Close_Yesterday'] = df['Close'].shift(1)
df = df.dropna()

#%% Merge dataframes
df = df.merge(df_fe, left_index=True, right_index=True)

# %% Split data
df = df.dropna()
#%%
#Shuffle the data
shuffled = df.sample(frac=1)
df = shuffled.copy()
#%% Normalize the data
scaler = StandardScaler()
scaler.fit(df.drop('Close', axis = 1))
df_scaled = pd.DataFrame(scaler.transform(df.drop('Close', axis = 1)), columns=df.drop('Close', axis = 1).columns)
df - df_scaled.copy()
#%%#80%train 20%test
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


#%%Plots
def make_fig(y_true,y_pred,index,model_name):
    trace0 = go.Scattergl(
    x=index,
    y=y_true,
    name='BTC Real',
    mode='markers+lines'
)
    trace1 = go.Scattergl(
        x=index,
        y=y_pred,
        name='BTC Previsto',
        mode='markers+lines'
    )
    layout = {
    'title': f'Predição do BTC usando {model_name}',
    'title_x': 0.5,
    'xaxis': {'title': 'Data'},
    'yaxis': {'title': 'Valor do BTC'},
    }

    fig = go.Figure(data=[trace0, trace1], layout=layout)
    fig.show()

#%% Model XGBoost
model = ModelXGboost(X_train_xgb, X_test_xgb, y_train_xgb)
model.fit()
result = model.predict()
predicao = result.copy()
real = y_test_xgb.values.copy()
mae = mean_absolute_error(predicao, real)
mse = mean_squared_error(predicao, real)
print('MAE: ', mae)
print('MSE: ', mse)
#Percentual de erro
percentual_dif = 0
for r,p in zip(predicao,real):
    percentual_dif += (abs(r-p)/r)
print('Percentual de erro do XGboost: +-', round(percentual_dif[0],2),"%")
make_fig(real.flatten(),predicao,index,'XGboost')
#%% 
