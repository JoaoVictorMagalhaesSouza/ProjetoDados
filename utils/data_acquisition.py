# Módulo para aquisição dos dados
import yfinance as yf
import numpy as np
#Lib para aquisição de dados via API 
import pandas_datareader as web
#https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

class DataAcquisition():

    def __init__(self, start_date = '2018-01-01'):
        self.start_date = start_date
    
    def get_data(self):
        df_bitcoin = web.get_data_yahoo('BTC-USD', start=self.start_date)
        return df_bitcoin