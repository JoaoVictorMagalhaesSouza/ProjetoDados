from tabnanny import verbose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import pandas as pd
#Import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
#Import ARIMA

class ModelLSTM():
    def __init__(self, X_train, x_test, y_train):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        self.model = Sequential()
        self.model.add(LSTM(46,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))    
        self.model.add(LSTM(64))        
        self.model.add(Dense(1))
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy', 'mae' ])
    
    def fit(self):
        print("X_train shape: ", self.X_train.shape)
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32)
    
    def predict(self):
        self.y_pred = self.model.predict(self.x_test).flatten()
        return self.y_pred
    
class ModelXGboost():
    def __init__(self, X_train, x_test, y_train):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        
        
    def fit(self):
        self.model = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=12, random_state=42,tree_method='gpu_hist',
        eval_metric='mae', gamma=0.5, reg_lambda = 0.6, reg_alpha=0.7)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred

class ModelCatboost():
    def __init__(self, X_train, x_test, y_train):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        
    def fit(self):
        self.model = CatBoostRegressor(iterations=1000, learning_rate=0.15, depth=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self):
        self.y_pred = self.model.predict(self.x_test)
        return self.y_pred