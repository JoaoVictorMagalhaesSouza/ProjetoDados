import pandas as pd
class DataPreparation():
    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()

    def remove_outliers(self):
        pass

    def normalize_data(self, normalization_type = 'std'):
        """
            Normalização de dados via média e desvio padrão
        """
        target = 'Close'
        if (normalization_type == 'std'):
            df_target_droped = self.data.drop(columns={target})
            self.data[df_target_droped.columns] =  (df_target_droped - df_target_droped.mean()) / df_target_droped.std()

        elif (normalization_type == 'min_max'):
            df_target_droped = self.data.drop(columns={target})
            self.data[df_target_droped.columns] =  (df_target_droped - df_target_droped.min()) / (df_target_droped.max() - df_target_droped.min())

        return self.data.copy()