from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
import os
from datetime import datetime
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class PatientInput(BaseModel):
    """Input schema for patient data"""
    age: float = Field(..., ge=0, le=120, description="Patient's age")
    gender: str = Field(..., description="Gender (Male/Female)")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index")
    smoker: str = Field(..., description="Smoking status (Yes/No)")
    chronic_cond: str = Field(..., description="Chronic condition status (Yes/No)")
    injury_type: str = Field(..., description="Type of injury")
    referral_source: str = Field(..., description="Source of referral")
    insurance_type: str = Field(..., description="Type of insurance")
    n_sessions: int = Field(..., ge=0, description="Number of sessions attended")
    avg_session_duration: float = Field(..., ge=0, description="Average session duration in minutes")
    first_week: int = Field(default=0, ge=0, description="Sessions in first week")
    last_week: int = Field(default=0, ge=0, description="Sessions in last week")
    mean_pain: float = Field(..., ge=0, le=10, description="Mean pain score (0-10)")
    mean_pain_delta: float = Field(..., description="Change in pain score")
    home_adherence_mean: float = Field(..., ge=0, le=100, description="Mean home exercise adherence percentage")
    satisfaction_mean: float = Field(..., ge=0, le=5, description="Mean patient satisfaction score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "gender": "Female",
                "bmi": 24.5,
                "smoker": "No",
                "chronic_cond": "No",
                "injury_type": "Knee Injury",
                "referral_source": "Doctor",
                "insurance_type": "Private",
                "n_sessions": 8,
                "avg_session_duration": 45.0,
                "first_week": 2,
                "last_week": 1,
                "mean_pain": 4.5,
                "mean_pain_delta": -1.2,
                "home_adherence_mean": 75.0,
                "satisfaction_mean": 4.2
            }
        }
    )

class DropoutPredictionResponse(BaseModel):
    """Response model for dropout prediction"""
    patient_id: str
    dropout_probability: float = Field(..., ge=0, le=1, description="Dropout probability (0-1)")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")
    recommendations: List[str] = Field(..., description="List of recommendations")

class SegmentationResponse(BaseModel):
    """Response model for patient segmentation"""
    patient_id: str
    cluster_id: int = Field(..., description="Cluster ID")
    cluster_name: str = Field(..., description="Cluster name/description")
    characteristics: List[str] = Field(..., description="Cluster characteristics")

class AdherenceResponse(BaseModel):
    """Response model for adherence prediction"""
    patient_id: str
    adherence_level: str = Field(..., description="Adherence level (Low/Medium/High)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Adherence recommendations")

class ComprehensiveResponse(BaseModel):
    """Response model for comprehensive predictions"""
    patient_id: str
    predictions: Dict[str, Any] = Field(..., description="All prediction results")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
    available_models: List[str]
    model_info: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchPatientInput(BaseModel):
    """Input for batch predictions"""
    patients: List[PatientInput]
    prediction_types: List[str] = Field(
        default=["dropout", "segmentation", "adherence"],
        description="Types of predictions to run"
    )

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_patients: int
    successful_predictions: int
    failed_predictions: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]] = Field(default_factory=list)

# ML Prediction Class
class MedoptixPredictor:
    """Handles all ML predictions for Medoptix"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self) -> bool:
        """Load all trained models"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            os.makedirs("models/dropout_prediction", exist_ok=True)
            os.makedirs("models/segmentation", exist_ok=True)
            os.makedirs("models/adherence_forecasting", exist_ok=True)
            
            # Dropout prediction models
            dropout_path = "models/dropout_prediction/"
            if os.path.exists(f"{dropout_path}medoptix_dropout_prediction_preprocessor.pkl"):
                self.models["dropout_preprocessor"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_preprocessor.pkl")
                self.models["dropout_model"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_model.pkl")
                self.models["dropout_feature_names"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_feature_names.pkl")
                self.models["dropout_columns"] = joblib.load(f"{dropout_path}medoptix_dropout_prediction_columns.pkl")
                logger.info("Dropout prediction models loaded successfully")
            else:
                logger.warning("Dropout prediction models not found")
            
            # Segmentation models
            seg_path = "models/segmentation/"
            if os.path.exists(f"{seg_path}medoptix_segmentation_preprocessor.pkl"):
                self.models["segmentation_preprocessor"] = joblib.load(f"{seg_path}medoptix_segmentation_preprocessor.pkl")
                self.models["segmentation_pca"] = joblib.load(f"{seg_path}medoptix_segmentation_pca.pkl")
                self.models["segmentation_kmeans"] = joblib.load(f"{seg_path}medoptix_segmentation_kmeans.pkl")
                logger.info("Segmentation models loaded successfully")
            else:
                logger.warning("Segmentation models not found")
            
            # Adherence forecasting models
            adh_path = "models/adherence_forecasting/"
            if os.path.exists(f"{adh_path}medoptix_adherence_forecasting_preprocessor.pkl"):
                self.models["adherence_preprocessor"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_preprocessor.pkl")
                self.models["adherence_model"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_model.pkl")
                self.models["adherence_label_encoder"] = joblib.load(f"{adh_path}medoptix_adherence_forecasting_label_encoder.pkl")
                logger.info("Adherence forecasting models loaded successfully")
            else:
                logger.warning("Adherence forecasting models not found")
            
            logger.info(f"Available models: {list(self.models.keys())}")
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_dropout(self, patient_data: Dict[str, Any]) -> Tuple[float, int, str, List[str]]:
        """Predict dropout probability and return recommendations"""
        try:
            if "dropout_model" not in self.models:
                raise ValueError("Dropout model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess and predict
            X_processed = self.models["dropout_preprocessor"].transform(df)
            dropout_prob_raw = self.models["dropout_model"].predict_proba(X_processed)[0, 1]
            dropout_prob = float(dropout_prob_raw)
            
            # Get risk level and recommendations
            risk_level, recommendations = self._get_dropout_risk_recommendations(dropout_prob)
            
            return dropout_prob, 0, risk_level, recommendations
            
        except Exception as e:
            logger.error(f"Error in dropout prediction: {str(e)}")
            return 0.5, 0, "Medium", ["Unable to generate recommendations due to prediction error"]
    
    def predict_segment(self, patient_data: Dict[str, Any]) -> Tuple[int, str, List[str]]:
        """Predict patient segment and return characteristics"""
        try:
            if "segmentation_kmeans" not in self.models:
                raise ValueError("Segmentation model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess
            X_processed = self.models["segmentation_preprocessor"].transform(df)
            
            # Apply PCA
            X_reduced = self.models["segmentation_pca"].transform(X_processed)
            
            # Predict cluster
            cluster = self.models["segmentation_kmeans"].predict(X_reduced)[0]
            
            # Get cluster characteristics
            cluster_name, characteristics = self._get_cluster_characteristics(cluster)
            
            return int(cluster), cluster_name, characteristics
            
        except Exception as e:
            logger.error(f"Error in segmentation prediction: {str(e)}")
            return 0, "Unknown", ["Unable to determine patient segment"]
    
    def predict_adherence(self, patient_data: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Predict adherence level and return recommendations"""
        try:
            if "adherence_model" not in self.models:
                raise ValueError("Adherence model not loaded")
            
            # Convert to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Preprocess and predict
            X_processed = self.models["adherence_preprocessor"].transform(df)
            pred_encoded = self.models["adherence_model"].predict(X_processed)[0]
            pred_proba = self.models["adherence_model"].predict_proba(X_processed)[0]
            
            # Decode prediction
            adherence_level = self.models["adherence_label_encoder"].inverse_transform([pred_encoded])[0]
            confidence = float(max(pred_proba))
            
            # Get recommendations
            recommendations = self._get_adherence_recommendations(adherence_level)
            
            return adherence_level, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Error in adherence prediction: {str(e)}")
            return "Medium", 0.33, ["Unable to generate adherence recommendations"]
    
    def _get_dropout_risk_recommendations(self, dropout_prob: float) -> Tuple[str, List[str]]:
        """Generate risk level and recommendations for dropout"""
        if dropout_prob > 0.7:
            return "High", [
                "Schedule immediate follow-up call within 24 hours",
                "Assign dedicated support specialist",
                "Offer flexible scheduling options",
                "Consider telehealth sessions",
                "Implement motivational interviewing techniques"
            ]
        elif dropout_prob > 0.3:
            return "Medium", [
                "Send weekly check-in messages",
                "Monitor attendance closely",
                "Offer additional support if needed",
                "Provide educational materials about treatment benefits",
                "Consider group therapy sessions"
            ]
        else:
            return "Low", [
                "Continue standard care procedures",
                "Maintain regular check-ins",
                "Celebrate progress milestones",
                "Provide positive reinforcement"
            ]
    
    def _get_cluster_characteristics(self, cluster: int) -> Tuple[str, List[str]]:
        """Get characteristics for each cluster"""
        cluster_info = {
            0: ("High Engagement", [
                "High session attendance",
                "Good home exercise adherence", 
                "Positive treatment outcomes",
                "Low dropout risk"
            ]),
            1: ("Moderate Engagement", [
                "Average session attendance",
                "Variable home exercise adherence",
                "Mixed treatment outcomes",
                "Moderate dropout risk"
            ]),
            2: ("Low Engagement", [
                "Poor session attendance",
                "Low home exercise adherence",
                "Concerning treatment outcomes",
                "High dropout risk"
            ])
        }
        
        return cluster_info.get(cluster, ("Unknown", ["Unable to determine characteristics"]))
    
    def _get_adherence_recommendations(self, adherence_level: str) -> List[str]:
        """Generate recommendations based on adherence level"""
        recommendations = {
            "High": [
                "Maintain current engagement strategies",
                "Use as peer mentor for other patients",
                "Consider advanced treatment protocols",
                "Schedule less frequent check-ins"
            ],
            "Medium": [
                "Provide additional motivation techniques",
                "Send reminder notifications",
                "Offer flexible exercise options",
                "Monitor progress more closely"
            ],
            "Low": [
                "Implement intensive support program",
                "Schedule frequent check-ins",
                "Simplify exercise protocols",
                "Consider motivational interviewing",
                "Address barriers to adherence"
            ]
        }
        
        return recommendations.get(adherence_level, ["Standard care recommendations"])
    
    def get_comprehensive_prediction(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get all predictions for a patient"""
        try:
            results = {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "predictions": {}
            }
            
            # Dropout prediction
            if "dropout_model" in self.models:
                dropout_prob, _, dropout_risk, dropout_recs = self.predict_dropout(patient_data)
                results["predictions"]["dropout"] = {
                    "probability": dropout_prob,
                    "risk_level": dropout_risk,
                    "recommendations": dropout_recs
                }
            
            # Segmentation prediction
            if "segmentation_kmeans" in self.models:
                cluster_id, cluster_name, cluster_chars = self.predict_segment(patient_data)
                results["predictions"]["segmentation"] = {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "characteristics": cluster_chars
                }
            
            # Adherence prediction
            if "adherence_model" in self.models:
                adh_level, confidence, adh_recs = self.predict_adherence(patient_data)
                results["predictions"]["adherence"] = {
                    "level": adh_level,
                    "confidence": confidence,
                    "recommendations": adh_recs
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive prediction: {str(e)}")
            return {
                "patient_id": patient_data.get("patient_id", "unknown"),
                "error": str(e),
                "predictions": {}
            }

# Utility functions
def generate_patient_id(patient_data: Dict[str, Any]) -> str:
    """Generate a unique patient ID based on input data"""
    data_str = str(sorted(patient_data.items()))
    hash_object = hashlib.md5(data_str.encode())
    return f"pt_{hash_object.hexdigest()[:8]}"

def get_predictor():
    """Dependency to get predictor instance with error handling"""
    if not predictor.models:
        raise HTTPException(status_code=503, detail="ML models not loaded. Please check server status.")
    return predictor

# Global instances
predictor = MedoptixPredictor()

# Initialize FastAPI app
app = FastAPI(
    title="MedOptix AI Platform",
    description="AI-powered patient analytics for physical therapy - dropout prediction, patient segmentation, and adherence forecasting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and initialize services on startup"""
    logger.info("Starting MedOptix API...")
    success = predictor.load_models()
    if not success:
        logger.error("Failed to load ML models")
    else:
        logger.info("ML models loaded successfully")

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint to verify API status and model availability"""
    models_loaded = bool(predictor.models)
    
    model_info = {
        "total_models": len(predictor.models),
        "dropout_model_available": "dropout_model" in predictor.models,
        "segmentation_model_available": "segmentation_kmeans" in predictor.models,
        "adherence_model_available": "adherence_model" in predictor.models,
    }
    
    return HealthResponse(
        status="Healthy" if models_loaded else "Degraded",
        models_loaded=models_loaded,
        available_models=list(predictor.models.keys()),
        model_info=model_info
    )

# Dropout prediction endpoint
@app.post("/predict/dropout", response_model=DropoutPredictionResponse, tags=["Predictions"])
async def predict_dropout(patient_data: PatientInput):
    """Predict patient dropout probability and risk level"""
    try:
        current_predictor = get_predictor()
        data_dict = patient_data.dict()
        dropout_prob, _, risk_level, recommendations = current_predictor.predict_dropout(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return DropoutPredictionResponse(
            patient_id=patient_id,
            dropout_probability=dropout_prob,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in dropout prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Patient segmentation endpoint
@app.post("/predict/segmentation", response_model=SegmentationResponse, tags=["Predictions"])
async def predict_segmentation(patient_data: PatientInput):
    """Predict patient segment/cluster based on characteristics"""
    try:
        current_predictor = get_predictor()
        data_dict = patient_data.dict()
        cluster_id, cluster_name, characteristics = current_predictor.predict_segment(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return SegmentationResponse(
            patient_id=patient_id,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            characteristics=characteristics
        )
        
    except Exception as e:
        logger.error(f"Error in segmentation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

# Adherence prediction endpoint
@app.post("/predict/adherence", response_model=AdherenceResponse, tags=["Predictions"])
async def predict_adherence(patient_data: PatientInput):
    """Predict patient adherence level"""
    try:
        current_predictor = get_predictor()
        data_dict = patient_data.dict()
        adherence_level, confidence, recommendations = current_predictor.predict_adherence(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return AdherenceResponse(
            patient_id=patient_id,
            adherence_level=adherence_level,
            confidence=confidence,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in adherence prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Adherence prediction failed: {str(e)}")

# Comprehensive prediction endpoint
@app.post("/predict/comprehensive", response_model=ComprehensiveResponse, tags=["Predictions"])
async def predict_comprehensive(patient_data: PatientInput):
    """Get all available predictions for a patient"""
    try:
        current_predictor = get_predictor()
        data_dict = patient_data.dict()
        results = current_predictor.get_comprehensive_prediction(data_dict)
        
        patient_id = generate_patient_id(data_dict)
        
        return ComprehensiveResponse(
            patient_id=patient_id,
            predictions=results["predictions"]
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(batch_input: BatchPatientInput):
    """Run predictions on multiple patients"""
    try:
        current_predictor = get_predictor()
        results = []
        errors = []
        successful = 0
        
        for i, patient_data in enumerate(batch_input.patients):
            try:
                data_dict = patient_data.dict()
                patient_id = generate_patient_id(data_dict)
                
                patient_results = {"patient_id": patient_id}
                
                # Run requested predictions
                if "dropout" in batch_input.prediction_types:
                    dropout_prob, _, risk_level, recs = current_predictor.predict_dropout(data_dict)
                    patient_results["dropout"] = {
                        "probability": dropout_prob,
                        "risk_level": risk_level,
                        "recommendations": recs
                    }
                
                if "segmentation" in batch_input.prediction_types:
                    cluster_id, cluster_name, chars = current_predictor.predict_segment(data_dict)
                    patient_results["segmentation"] = {
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                        "characteristics": chars
                    }
                
                if "adherence" in batch_input.prediction_types:
                    adh_level, confidence, adh_recs = current_predictor.predict_adherence(data_dict)
                    patient_results["adherence"] = {
                        "level": adh_level,
                        "confidence": confidence,
                        "recommendations": adh_recs
                    }
                
                results.append(patient_results)
                successful += 1
                
            except Exception as e:
                errors.append({
                    "patient_index": i,
                    "error": str(e)
                })
        
        return BatchPredictionResponse(
            total_patients=len(batch_input.patients),
            successful_predictions=successful,
            failed_predictions=len(errors),
            results=results,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Model information endpoints
@app.get("/models/info", tags=["System"])
async def get_model_info():
    """Get detailed information about loaded models"""
    try:
        model_info = {
            "loaded_models": list(predictor.models.keys()),
            "model_details": {}
        }
        
        # Add model-specific information
        if "dropout_model" in predictor.models:
            model_info["model_details"]["dropout"] = {
                "type": str(type(predictor.models["dropout_model"]).__name__),
                "features": len(predictor.models.get("dropout_feature_names", [])),
            }
        
        if "segmentation_kmeans" in predictor.models:
            model_info["model_details"]["segmentation"] = {
                "type": "KMeans",
                "n_clusters": getattr(predictor.models["segmentation_kmeans"], 'n_clusters', 'unknown')
            }
        
        if "adherence_model" in predictor.models:
            model_info["model_details"]["adherence"] = {
                "type": str(type(predictor.models["adherence_model"]).__name__),
            }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "MedOptix AI Platform API",
        "version": "1.0.0",
        "description": "AI-powered patient analytics for physical therapy",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "dropout_prediction": "/predict/dropout",
            "segmentation": "/predict/segmentation",
            "adherence": "/predict/adherence",
            "comprehensive": "/predict/comprehensive",
            "batch": "/predict/batch"
        }
    }