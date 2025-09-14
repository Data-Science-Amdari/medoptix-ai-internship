import numpy as np
import pandas as pd
from src.prediction import train_dropout_model, predict_dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def test_dropout_prediction():
    # Dummy data
    X = pd.DataFrame(np.random.rand(200, 10))
    y = np.random.randint(0, 2, 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_dropout_model(X_train, y_train)
    probs = predict_dropout(model, X_test)
    assert np.all((probs >= 0) & (probs <= 1)), 'Probabilities out of bounds'
    auc = roc_auc_score(y_test, probs)
    assert auc > 0.6, f'ROC AUC too low: {auc}' 