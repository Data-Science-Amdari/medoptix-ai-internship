
# forecasting.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def prepare_forecasting_data(df, target_column, feature_columns):
    """Prepares data for forecasting."""
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_forecasting_model(X_train, y_train):
    """Train a linear regression model (can be replaced with advanced models)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_forecasting_model(model, X_test, y_test):
    """Evaluate a forecasting model using MAE and RMSE."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return predictions, mae, rmse
