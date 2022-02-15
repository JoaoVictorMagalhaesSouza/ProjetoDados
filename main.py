#%% Imports
import pandas as pd
import numpy as np
import plotly.express as px

#Defining the working directory for internal functions
import os, sys
dir_import = os.getcwd()
sys.path.insert(0,dir_import+'/../')
from utils.data_acquisition import DataAcquisition
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering

#%% Data Acquisition
df = DataAcquisition().get_data()
# %% Data Preparation
df = DataPreparation(df).normalize_data()

#%% Feature Engineering
df_fe = FeatureEngineering(df).pipeline_feat_eng()