#%% Imports
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns

#Defining the working directory for internal functions
import os, sys
dir_import = os.getcwd()
sys.path.insert(0,dir_import+'/../')
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
from utils.correlation_analysis import CorrelationAnalysis

#%% Data Acquisition
df = DataAcquisition().get_data()
# %% Data Preparation
df = DataPreparation(df).normalize_data()

#%% Feature Engineering
df_fe = FeatureEngineering(df).pipeline_feat_eng()
# %% Merge dataframes
df = df.merge(df_fe, left_index=True, right_index=True)
# %% Correlation Analysis
df_corr = CorrelationAnalysis(df).melhores_variaveis()

# %%
sns.heatmap(df_corr, annot=True, linewidths=.5)

# %%
