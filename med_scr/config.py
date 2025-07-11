
# config.py

# Paths
AWS_ENV_PATH = "c:/Users/kaoth/OneDrive/Desktop/Data Science Lectures/Machine Learning/Amdari Internship/Project_6/aws.env"
AIVEN_ENV_PATH = "C:/Users/kaoth/OneDrive/Desktop/Data Science Lectures/Machine Learning/Amdari Internship/Project_6/aiven.env"

# Database tables
TABLES = [
    "patients", "sessions", "feedback",
    "clinics", "interventions", "dropout_flags"
]

# Prediction threshold
DEFAULT_THRESHOLD = 0.25

# Forecasting configuration
FORECASTING_TARGET = "missed_sessions"
FEATURE_COLUMNS = ["weeks_enrolled", "total_sessions", "completed_sessions"]
