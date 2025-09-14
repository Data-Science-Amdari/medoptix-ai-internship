from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field
import os
from app.prediction import MedoptixPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = MedoptixPredictor()


# Lifespan context manager (replaces deprecated startup/shutdown events)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Medoptix API...")
    success = predictor.load_models()
    if not success:
        logger.error("Failed to load models")
        raise RuntimeError("Failed to load models")
    logger.info("Models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medoptix API...")


# Step 1: Initialize FastAPI app with lifespan
app = FastAPI(
    title="Medoptix API",
    description="AI-powered dropout prediction and patient clustering for physical therapy",
    version="1.0.0",
    lifespan=lifespan
)


# Step 2: Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Step 3: Pydantic models for input/output data
class PatientData(BaseModel):
    # Patient basic information
    age: float = Field(..., le=120, ge=0, description="Patient's age")
    gender: str = Field(..., description="male or female")
    bmi: float = Field(..., le=100, ge=0, description="Body Mass Index")
    smoker: str = Field(..., description="smoker or non-smoker")
    chronic_cond: str = Field(..., description="chronic condition or not")
    injury_type: str = Field(..., description="type of injury")
    referral_source: str = Field(..., description="source of referral")  # Fixed typo
    insurance_type: str = Field(..., description="type of insurance")

    # Session information
    n_sessions: int = Field(..., ge=0, description="Number of sessions attended")
    avg_session_duration: float = Field(..., ge=0, description="Average session duration in minutes")
    first_week: int = Field(..., ge=0, description="Number of sessions in the first week")
    last_week: int = Field(..., ge=0, description="Number of sessions in the last week")
    mean_pain: float = Field(..., description="Mean pain score")
    mean_pain_delta: float = Field(..., description="Change in mean pain score")
    home_adherence_mean: float = Field(..., ge=0, description="Mean home exercise adherence score")
    satisfaction_mean: float = Field(..., ge=0, description="Mean patient satisfaction score")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "gender": "Female",
                "bmi": 22.5,
                "smoker": "No",
                "chronic_cond": "No",
                "injury_type": "Knee Injury",
                "referral_source": "Doctor",  # Fixed typo
                "insurance_type": "Private",
                "n_sessions": 10,
                "avg_session_duration": 45.0,
                "first_week": 3,
                "last_week": 2,
                "mean_pain": 5.0,
                "mean_pain_delta": -1.0,
                "home_adherence_mean": 80.0,
                "satisfaction_mean": 4.5
            }
        }


class DropoutPredictionResponse(BaseModel):
    patient_id: str 
    dropout_probability: float
    risk_level: str
    recommendations: List[str]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]
    model_info: Dict[str, Any]


# Step 4: Root endpoint
@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Medoptix API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Step 5: Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API status and model availability
    """
    models_loaded = bool(predictor.models)
    model_info = {}

    try: 
        feature_names = predictor.models.get("feature_names")
        column_names = predictor.models.get("columns")

        if feature_names is not None:
            if isinstance(feature_names, np.ndarray):
                feature_names = feature_names.tolist()
            elif not isinstance(feature_names, list):
                feature_names = [str(feature_names)]
        
        if column_names is not None:
            if isinstance(column_names, np.ndarray):
                column_names = column_names.tolist()
            elif not isinstance(column_names, list):
                column_names = [str(column_names)]

        model_info = {
            "feature_names": feature_names or [],
            "column_names": column_names or [],
            "num_features": len(feature_names) if feature_names else 0,
            "num_columns": len(column_names) if column_names else 0
        }
                
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        model_info = {"error": "Could not retrieve model information"}
        
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        available_models=list(predictor.models.keys()) if predictor.models else [],
        model_info=model_info
    )


def prepare_patient_data(patient: PatientData) -> pd.DataFrame:
    """Convert patient data to DataFrame with columns in the order expected by the model"""
    try:
        columns = predictor.models.get("columns", [])
        patient_dict = patient.dict()

        # Create DataFrame with the same columns as the model expects
        df_data = {}
        for col in columns:
            if col in patient_dict:
                df_data[col] = [patient_dict[col]]
            else:
                # Handle column name mismatches (e.g., referral vs refferal)
                if col == "refferal_source" and "referral_source" in patient_dict:
                    df_data[col] = [patient_dict["referral_source"]]
                # Provide default values for missing columns
                elif col in ['n_sessions', 'first_week', 'last_week']:
                    df_data[col] = [0]
                elif col in ['avg_session_duration', 'mean_pain', 'mean_pain_delta', 'home_adherence_mean', 'satisfaction_mean']:
                    df_data[col] = [0.0]
                else:
                    df_data[col] = ["Unknown"]

        return pd.DataFrame(df_data, columns=columns)
    
    except Exception as e:
        logger.error(f"Error preparing patient data: {str(e)}")
        raise


# Step 6: Prediction endpoint
@app.post("/predict/dropout", response_model=DropoutPredictionResponse)
async def predict_dropout(patient_data: PatientData):
    """
    Predict dropout probability and provide recommendations based on patient data
    """
    try:
        if not predictor.models:
            raise HTTPException(status_code=500, detail="Models not loaded")

        # Make prediction
        dropout_prob, _, risk_level, recommendations = predictor.predict_dropout(patient_data.dict())

        # Generate patient ID
        patient_id = f"pt_{hash(str(patient_data.dict())) % 10000:05d}"

        return DropoutPredictionResponse(
            patient_id=patient_id,
            dropout_probability=float(dropout_prob),  # Ensure it's a float
            risk_level=risk_level,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Error in dropout prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Step 7: Additional utility endpoints
@app.get("/models/info")
async def get_model_info():
    """Get detailed information about loaded models"""
    if not predictor.models:
        raise HTTPException(status_code=404, detail="No models loaded")
    
    return {
        "loaded_models": list(predictor.models.keys()),
        "model_details": {
            key: {
                "type": str(type(value).__name__),
                "size": len(str(value)) if hasattr(value, '__len__') else "N/A"
            }
            for key, value in predictor.models.items()
            if value is not None
        }
    }


# Step 8: Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )