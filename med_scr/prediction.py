# med_scr/prediction.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import shap
import joblib


def train_model(X_train, y_train, model_name="random_forest", tune=False):
    """
    Train a classification model. Optionally performs hyperparameter tuning for Random Forest.
    """
    if model_name == "random_forest":
        model = RandomForestClassifier(random_state=42)

        if tune:
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
                "max_features": ["sqrt", "log2", None],
            }

            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                scoring="f1",
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print("✅ Best Random Forest Params:", search.best_params_)

    elif model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)

    elif model_name == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    else:
        raise ValueError(f"❌ Unknown model type: {model_name}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")

    if y_proba is not None:
        roc_score = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC Score: {roc_score:.3f}")
        return y_pred, y_proba, roc_score

    return y_pred, None, None


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model using a custom threshold for classification.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n📊 Evaluation at Threshold = {threshold:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

    return y_pred, y_proba


def tune_threshold(model, X_test, y_test, thresholds=np.arange(0.1, 0.9, 0.05)):
    """
    Tune classification threshold to maximize F1 Score.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n🔧 Threshold Tuning")
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_test, y_pred)
        print(f"Threshold: {t:.2f} - F1 Score: {score:.3f}")


def explain_model(model, X_train, X_test, feature_names):
    """
    Generate SHAP summary plot for a model.
    """
    if "RandomForest" in str(type(model)) or "XGB" in str(type(model)):
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test, check_additivity=False)
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names)


def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation.
    """
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv)
    print(f"\nCross-validated F1 score (cv={cv}): {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
    return scores


def save_model(model, path):
    """
    Save a trained model to disk.
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load a saved model from disk.
    """
    return joblib.load(path)
