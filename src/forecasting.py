import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib

def train_forecasting_model(X_train, y_train, params=None):
    model = RandomForestRegressor(**(params or {}))
    model.fit(X_train, y_train)
    return model

def predict_adherence(model, X):
    return model.predict(X)

def evaluate_forecast(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    residuals = y_true - y_pred
    return mape, residuals

def save_model(model, path):
    joblib.dump(model, path) 