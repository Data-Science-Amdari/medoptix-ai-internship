
# med_tests/test_segmentation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from segmentation import preprocess_segmentation_data, run_clustering
import pandas as pd

# Dummy mock data
@pytest.fixture
def mock_patient_data():
    data = {
        "age": [25, 40, 60, 30],
        "sessions_attended": [5, 15, 2, 10],
        "intervention_score": [3.5, 4.2, 2.1, 3.8],
    }
    return pd.DataFrame(data)

def test_preprocessing_returns_dataframe(mock_patient_data):
    X_scaled = preprocess_segmentation_data(mock_patient_data)
    assert isinstance(X_scaled, pd.DataFrame)
    assert not X_scaled.empty
    assert X_scaled.shape[0] == mock_patient_data.shape[0]

def test_clustering_assigns_labels(mock_patient_data):
    X_scaled = preprocess_segmentation_data(mock_patient_data)
    clustered_df, _ = run_clustering(X_scaled, mock_patient_data)
    assert "cluster" in clustered_df.columns
    assert clustered_df["cluster"].nunique() > 0
