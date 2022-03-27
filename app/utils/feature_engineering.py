import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class FeatureEngineering():
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.df_fe = pd.DataFrame()
        self.target = 'Close'

    def derivada(self):        
        for column in self.data.columns:
            if column != self.target:
                 self.df_fe[f'{column}_derivative'] = self.data[column].diff()

    def integral(self):
        for column in self.data.columns:
            if column != self.target:
                #Integral numa janela de 3 dias.
                self.df_fe[f'{column}_integral'] = self.data[column].rolling(3).sum()

    def momentos_estatisticos(self):
        for column in self.data.columns:
            if column != self.target:
                #Média móvel e desvio padrão de 3 dias.
                self.df_fe[f'{column}_moving_average'] = self.data[column].rolling(3).mean()
                self.df_fe[f'{column}_std'] = self.data[column].rolling(3).std()
    
    def combinacoes_polinomiais(self):
        df_poly = PolynomialFeatures(2)
        cols = self.data.columns
        df_poly = pd.DataFrame(df_poly.fit_transform(self.data[cols]))
        qtde_colunas = len(df_poly.columns)
        df_poly = df_poly.drop(columns=[x for x in range (len(cols)+1)])
        nome_novas_colunas = []
        nao_vistadas = list(cols.copy())
        for coluna in cols:
            atual = coluna
            nome_novas_colunas.append(f'{coluna}^2')
            for _ in nao_vistadas:
                if (_ != atual):
                    nome_novas_colunas.append(f'{coluna}*{_}')
            
            nao_vistadas.remove(coluna)
        
        nome_velhas_colunas = [x for x in range(len(cols)+1,qtde_colunas)]
        for i in range(nome_velhas_colunas[0],nome_velhas_colunas[-1]+1):
            df_poly = df_poly.rename(columns={i:nome_novas_colunas[i-nome_velhas_colunas[0]]})

        for col in df_poly.columns:
            self.df_fe[col] = df_poly[col].values
    
    def difference(self):
        df_final = pd.DataFrame()
        df_final['high-low'] = self.data['High'] - self.data['Low']
        df_final['high-close'] = self.data['High'] - self.data['Close']
        df_final['low-close'] = self.data['Low'] - self.data['Close']
        df_final['close-open'] = self.data['Close'] - self.data['Open']
        df_final['high-open'] = self.data['High'] - self.data['Open']
        df_final['low-open'] = self.data['Low'] - self.data['Open']

        self.df_fe = self.df_fe.merge(df_final, left_index=True, right_index=True)



    def pipeline_feat_eng(self):
        self.derivada()
        self.integral()
        self.momentos_estatisticos()
        self.combinacoes_polinomiais()
        self.difference()
        return self.df_fe.copy()