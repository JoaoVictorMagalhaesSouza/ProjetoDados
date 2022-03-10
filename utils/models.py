from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import pandas as pd
#Import LSTM
from tensorflow.python.keras.layers import Input, Dense, Dense, Dropout, LSTM, Bidirectional
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
import numpy as np
#Import ARIMA
from statsmodels.tsa.arima.model import ARIMA
#Construção do modelo ARIMA
class ModelARIMA():
    def __init__(self,df):
        self.df = df
        self.target = self.df['Close'].values          
        self.model = ARIMA(self.target, order=(7, 1, 1))

    def fit(self):
        self.fit = self.model.fit()
    
    def predict(self):
        self.y_pred = self.model.predict(start=len(self.X_train), end=len(self.X_train)+len(self.x_test)-1, dynamic=False)
        return self.y_pred
    def summary(self):
        return self.fit.summary()

class ModelLSTM():
    def __init__(self, X_train, x_test, y_train, y_test):
        self.X_train = X_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = Sequential()
        self.model.add(LSTM(64,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    
        self.model.add(LSTM(64))
        
        self.model.add(Dense(1))
        self.model.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
    
    def fit(self):
        print("X_train shape: ", self.X_train.shape)
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32)
    
    def predict(self):
        self.y_pred = self.model.predict(self.x_test).flatten()
        return self.y_pred
    
