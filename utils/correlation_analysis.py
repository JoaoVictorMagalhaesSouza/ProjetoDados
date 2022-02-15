import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

class CorrelationAnalysis():
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.df_corr = pd.DataFrame()
        self.target = 'Close'
        self.corr_threshold = 0.6
        
    def calcula_correlacao_target(self):
        self.df_corr = self.data.corr(method='pearson').abs()[self.target]
        self.df_corr = self.df_corr[(self.df_corr.index != self.target) & (self.df_corr > self.corr_threshold)]
        #Retorna uma lista com os nomes das features que deram correlação maior que o threshold
        return list(self.df_corr.index)

    def melhores_variaveis(self):
        df_corr_fe = pd.DataFrame()
        df_corr_fe = self.data[self.calcula_correlacao_target()].corr(method='pearson').abs()
        return df_corr_fe
