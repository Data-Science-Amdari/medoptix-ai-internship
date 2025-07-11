
# med_tests/test_prediction.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import pandas as pd
from prediction import train_model, evaluate_with_threshold
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        "feature1": [0.2, 0.4, 0.6, 0.8, 1.0],
        "feature2": [1, 3, 5, 7, 9],
        "label": [0, 0, 1, 1, 0]
    })
    X = df[["feature1", "feature2"]]
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_train_model_returns_model(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_name="random_forest")
    assert hasattr(model, "predict")

def test_custom_threshold_evaluation(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_name="logistic_regression")
    y_pred, y_proba = evaluate_with_threshold(model, X_test, y_test, threshold=0.4)
    assert len(y_pred) == len(y_test)
