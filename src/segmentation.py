import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def fit_segmenter(X, n_clusters=4, pca_components=2, model_path=None):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_components)),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=42))
    ])
    pipeline.fit(X)
    if model_path:
        joblib.dump(pipeline, model_path)
    return pipeline

def predict_cluster(pipeline, X):
    return pipeline.predict(X)

def evaluate_clustering(X, labels):
    score = silhouette_score(X, labels)
    return score 