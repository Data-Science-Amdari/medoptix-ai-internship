import os
import pandas as pd
import logging
from pathlib import Path

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DATA_DIR = Path('Datasets/raw')
PROCESSED_DATA_DIR = Path('Datasets/processed_feature_engineering')


def clean_column_names(df):
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    return df

def handle_missing(df):
    # Simple missing value handling: fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '')
    return df

def validate_data(df, critical_cols=None):
    if critical_cols:
        missing = df[critical_cols].isnull().sum()
        if missing.any():
            logging.warning(f'Missing values in critical columns: {missing[missing > 0]}')
    return df

def etl():
    setup_logger()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file in RAW_DATA_DIR.glob('*.csv'):
        logging.info(f'Processing {file.name}')
        df = pd.read_csv(file)
        df = clean_column_names(df)
        df = handle_missing(df)
        df = validate_data(df)
        out_path = PROCESSED_DATA_DIR / file.name.replace('.csv', '_processed.csv')
        df.to_csv(out_path, index=False)
        logging.info(f'Saved processed file to {out_path}')

if __name__ == '__main__':
    etl() 