
# src/etl.py

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import boto3


# --- Load Environment Variables ---
def load_env_files():
    load_dotenv(dotenv_path="C:/Users/kaoth/OneDrive/Desktop/Data Science Lectures/Machine Learning/Amdari Internship/Project_6/aws.env")
    load_dotenv(dotenv_path="C:/Users/kaoth/OneDrive/Desktop/Data Science Lectures/Machine Learning/Amdari Internship/Project_6/aiven.env")


# --- PostgreSQL (Aiven) ---
def get_db_engine():
    """Create and return SQLAlchemy engine for PostgreSQL."""
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(db_url)


def load_tables(engine):
    """Load tables from PostgreSQL into pandas DataFrames."""
    queries = {
        "patients": "SELECT * FROM patients",
        "sessions": "SELECT * FROM sessions",
        "feedback": "SELECT * FROM feedback",
        "clinics": "SELECT * FROM clinics",
        "interventions": "SELECT * FROM interventions",
        "dropout": "SELECT * FROM dropout_flags",
    }

    dfs = {}
    for name, query in queries.items():
        dfs[name] = pd.read_sql(query, engine)
        print(f"✅ Loaded '{name}' with shape {dfs[name].shape}")

    return dfs


# --- AWS S3 ---
def get_s3_client():
    """Initialize and return a boto3 S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def upload_to_s3(bucket, file_path, s3_key):
    """Upload local file to S3."""
    s3 = get_s3_client()
    s3.upload_file(file_path, bucket, s3_key)
    print(f"✅ Uploaded {file_path} to s3://{bucket}/{s3_key}")


# --- Data Cleaning (Optional placeholder) ---
def clean_data(df):
    """Basic placeholder for cleaning (customize later)."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


# --- Main Pipeline (if run standalone) ---
if __name__ == "__main__":
    load_env_files()
    engine = get_db_engine()
    dataframes = load_tables(engine)

    # Example usage
    patients_clean = clean_data(dataframes["patients"])
    print(patients_clean.head())
