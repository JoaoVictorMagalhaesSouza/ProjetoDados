from logging import exception
from os import remove
import pandas as pd
import numpy as np
import json
from pyodbc import connect
import pyodbc
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
def cria_conexão_banco() -> pyodbc.Connection:

    #Criando a conexão com o Banco de Dados
    host = '107.178.209.247'
    database = 'real_time_data'
    user = 'sqlserver'
    password = '1'

    conexao = connect(
            driver='{ODBC Driver 17 for SQL Server}',
            host=host,
            database=database,
            user=user,
            password=password
    )
    return conexao


def banco_de_dados_real_time():
    
    conexao = cria_conexão_banco()
    query = 'SELECT * FROM dados_tempo_real'
    try:    
        df_real_time = pd.read_sql(query, conexao)
    except exception as e:
        raise(f"Erro ao ler os dados do Banco de Dados em Tempo Real. Erro: {e}")

   
   
    return df_real_time
