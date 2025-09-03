import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib

def train_dropout_model(X_train, y_train, params=None):
    model = XGBClassifier(**(params or {}))
    model.fit(X_train, y_train)
    return model

def predict_dropout(model, X):
    return model.predict_proba(X)[:, 1]

def save_model(model, path):
    joblib.dump(model, path) 