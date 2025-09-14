import pandas as pd
from src.etl import etl, PROCESSED_DATA_DIR

def test_etl_output():
    etl()  # Run ETL
    df = pd.read_csv(PROCESSED_DATA_DIR / 'patients_processed.csv')
    assert df.shape[0] > 0, 'No rows in processed data'
    assert 'patient_id' in df.columns, 'Missing patient_id column'
    assert not df['patient_id'].isnull().any(), 'Missing values in patient_id' 