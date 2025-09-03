from pathlib import Path

RAW_DATA_DIR = Path('Datasets/raw')
PROCESSED_DATA_DIR = Path('Datasets/processed_feature_engineering')
SEGMENTATION_MODEL_PATH = Path('models/saved_models/segmentation_pipeline.joblib')
DROPOUT_MODEL_PATH = Path('models/saved_models/dropout_model.joblib')
FORECASTING_MODEL_PATH = Path('models/saved_models/forecasting_model.joblib')
FIGURES_DIR = Path('reports/figures') 