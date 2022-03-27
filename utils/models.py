
from xgboost import XGBRegressor

#Import ARIMA

class ModelXGboost():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        
    def fit(self):
        self.model = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=12, random_state=42,tree_method='gpu_hist',
        eval_metric='mae', gamma=0.5, reg_lambda = 0.6, reg_alpha=0.7)
        self.model.fit(self.X_train, self.y_train)
        
    def predict(self, X_test_xgb):
        self.y_pred = self.model.predict(X_test_xgb)
        return self.y_pred
    
    def get_booster(self):
        return self.model.get_booster()

