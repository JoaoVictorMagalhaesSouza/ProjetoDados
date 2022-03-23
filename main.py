#%% Imports
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
from utils.models import ModelLSTM, ModelXGboost

#%% Data Acquisition
df = DataAcquisition().get_data()
# %% Data Preparation
df = DataPreparation(df).normalize_data()

#%% Feature Engineering
df_fe = FeatureEngineering(df).pipeline_feat_eng()
#%% Merge dataframes
df = df.merge(df_fe, left_index=True, right_index=True)
#%% Correlation Analysis -  A fazer :

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

#%% Scaller
from sklearn.preprocessing import MinMaxScaler
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



#%% Model LSTM
model = ModelLSTM(X_train, X_test, y_train)
model.fit()
result = model.predict()

# Evaluate
predicao = output_scaler.inverse_transform(result.reshape(-1,1))
real = output_scaler.inverse_transform(y_test.reshape(-1,1))
mae = mean_absolute_error(predicao, real)
mse = mean_squared_error(predicao, real)
print('MAE: ', mae)
print('MSE: ', mse)
#Percentual de erro
percentual_dif = 0
for r,p in zip(predicao,real):
    percentual_dif += (abs(r-p)/r)
print('Percentual de erro da LSTM: +-', round(percentual_dif[0],2),"%")
# %% Model XGBoost
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

# %%Plot the results
pyplot.plot(real, label = 'Real')
pyplot.plot(predicao, label = 'Predicted')
pyplot.legend()
pyplot.show()
