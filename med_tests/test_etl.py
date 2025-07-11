# tests/test_etl.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from etl import load_env_files, get_db_engine, load_tables

@pytest.fixture(scope="module")
def setup_data():
    """Load environment and get engine and tables once per module."""
    load_env_files()
    engine = get_db_engine()
    dfs = load_tables(engine)
    return dfs

def test_connection(setup_data):
    """Check that data was loaded from the database."""
    dfs = setup_data
    assert isinstance(dfs, dict)
    assert "patients" in dfs
    assert "sessions" in dfs

def test_patients_df_not_empty(setup_data):
    """Ensure patients DataFrame is not empty."""
    patients_df = setup_data["patients"]
    assert not patients_df.empty
    assert patients_df.shape[0] > 0

def test_sessions_df_columns(setup_data):
    """Ensure sessions table contains expected columns."""
    sessions_df = setup_data["sessions"]
    expected_cols = {"patient_id", "session_date"}
    assert expected_cols.issubset(set(sessions_df.columns))
