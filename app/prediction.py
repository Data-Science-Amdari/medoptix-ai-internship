import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

class MedoptixPredictor:
    """ Helps handle all the ML prediction for Medoptix"""

    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self) -> bool:
        """ load all trained models"""
        try:
            model_path = "models/"
            self.models["feature_preprocessor"] = joblib.load(f"{model_path}ml_preprocessor.pkl")        
            self.models["dropout_model"] = joblib.load(f"{model_path}best_model_pipeline.pkl")        
            self.models["feature_names"] = joblib.load(f"{model_path}feature_names.pkl")
            self.models["columns"] = joblib.load(f"{model_path}prediction_columns.pkl") 

            logger.info("Models loaded successfully")
            logger.info(f"Available models: {list(self.models.keys())}")
            logger.info(f"feature names: {self.models['feature_names']}")
            logger.info(f"columns: {self.models['columns']}")
            return True   
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def _preprocess_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform patient data to match the EXACT format used during training
        All categorical columns must be converted to integers (label encoded)
        """
        logger.info(f"Starting preprocessing to match training pipeline...")
        
        processed_data = {}
        
        # 1. NUMERICAL FEATURES (used as-is during training)
        processed_data['n_sessions'] = patient_data.get('n_sessions', 0)
        processed_data['avg_session_duration'] = patient_data.get('avg_session_duration', 0.0)
        processed_data['satisfaction_mean'] = patient_data.get('satisfaction_mean', 0.0)
        processed_data['home_adherence_mean'] = patient_data.get('home_adherence_mean', 0.0)
        
        # 2. RAW INJURY_TYPE (gets one-hot encoded by the preprocessor)
        raw_injury_type = patient_data.get('injury_type', 'Knee').replace(' Injury', '').replace(' injury', '').strip().title()
        processed_data['injury_type'] = raw_injury_type
        
        # 3. CATEGORICAL FEATURES - CONVERT TO INTEGERS (same as training)
        
        # Gender (label encoded during training - alphabetical order typically)
        gender = patient_data.get('gender', 'Female')
        gender_mapping = {'Female': 0, 'Male': 1}  # Typical alphabetical encoding
        processed_data['gender'] = gender_mapping.get(gender, 0)
        
        # Smoker (label encoded during training)
        smoker = patient_data.get('smoker', 'No')
        smoker_mapping = {'No': 0, 'Yes': 1}  # Typical alphabetical encoding
        processed_data['smoker'] = smoker_mapping.get(smoker, 0)
        
        # Referral source (was simplified AND label encoded during training)
        referral_raw = patient_data.get('referral_source', 'Doctor')
        if referral_raw in ['Insurance', 'GP', 'Hospital']:
            referral_simplified = 'Professional'
        elif referral_raw == 'Self-Referral':
            referral_simplified = 'Self'
        else:
            referral_simplified = referral_raw  # Doctor, etc.
        
        # Map to integers (alphabetical order typically)
        referral_mapping = {'Doctor': 0, 'Professional': 1, 'Self': 2}
        processed_data['referral_source'] = referral_mapping.get(referral_simplified, 0)
        
        # Consent (was label encoded during training)
        processed_data['consent'] = 1  # Always 1 (assuming all patients consented)
        
        # Insurance type (was simplified AND label encoded during training)
        insurance_raw = patient_data.get('insurance_type', 'Private')
        if insurance_raw in ['Private-Premium', 'Private-Basic', 'Private-Top-Up']:
            insurance_simplified = 'Private'
        else:
            insurance_simplified = insurance_raw
        
        # Map to integers (alphabetical order typically)
        insurance_mapping = {'Medicare': 0, 'Private': 1, 'Public': 2}  # Common insurance types
        processed_data['insurance_type'] = insurance_mapping.get(insurance_simplified, 1)
        
        # BMI category (was already mapped to integers during training)
        bmi = patient_data.get('bmi', 25.0)
        if bmi < 18.5:
            bmi_category = 'Underweight'
        elif bmi < 25:
            bmi_category = 'Normal'
        elif bmi < 30:
            bmi_category = 'Overweight'
        else:
            bmi_category = 'Obese'
        
        # Map to the same integers used during training
        bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        processed_data['bmi_category'] = bmi_mapping[bmi_category]
        
        # Has chronic condition (was already converted to integers during training)
        chronic_cond = patient_data.get('chronic_cond', 'No')
        processed_data['has_chronic_cond'] = 1 if chronic_cond.lower() in ['yes', 'true', '1'] else 0
        
        logger.info(f"Processed data for training pipeline (all integers):")
        logger.info(f"  - Numerical: n_sessions={processed_data['n_sessions']}, avg_session_duration={processed_data['avg_session_duration']}")
        logger.info(f"  - Raw injury_type: {processed_data['injury_type']}")
        logger.info(f"  - Gender: {gender} -> {processed_data['gender']}")
        logger.info(f"  - Smoker: {smoker} -> {processed_data['smoker']}")
        logger.info(f"  - Referral: {referral_raw} -> {referral_simplified} -> {processed_data['referral_source']}")
        logger.info(f"  - Insurance: {insurance_raw} -> {insurance_simplified} -> {processed_data['insurance_type']}")
        logger.info(f"  - BMI category: {processed_data['bmi_category']}")
        logger.info(f"  - Has chronic condition: {processed_data['has_chronic_cond']}")
        
        return processed_data

    def predict_dropout(self, patient_data: Dict[str, Any]) -> Tuple[float, float, str, List[str]]:
        """
        Predict dropout probability and return all relevant information
        
        The dropout_model is a full pipeline: preprocessing -> SMOTE -> classifier
        So we only need to pass the raw DataFrame to it.
        """
        try:
            if not self.models:
                logger.error("Models not loaded. Cannot make predictions.")
                return 0.5, 0.0, "Medium", ["Unable to generate recommendations - models not loaded"]
            
            logger.info(f"=== STARTING PREDICTION ===")
            logger.info(f"Input injury_type: {patient_data.get('injury_type', 'NOT_PROVIDED')}")
            
            # Step 1: Preprocess the data to match training pipeline exactly  
            processed_data = self._preprocess_patient_data(patient_data)
            
            # Step 2: Convert to DataFrame with the exact columns used during training
            df = pd.DataFrame([processed_data])
            
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"Making prediction for patient with injury: {processed_data['injury_type']}")
            
            # Step 3: Use the FULL pipeline (dropout_model) directly on raw DataFrame
            # The dropout_model contains: preprocessing -> SMOTE -> classifier
            # So we don't need to apply feature_preprocessor separately
            dropout_proba_array = self.models["dropout_model"].predict_proba(df)
            dropout_prob_raw = dropout_proba_array[0, 1]
            dropout_prob = float(dropout_prob_raw)
            
            # Calculate confidence score
            confidence_score = float(max(dropout_proba_array[0]) - min(dropout_proba_array[0]))
            
            logger.info(f"Pipeline prediction successful: {dropout_prob:.3f} probability")

            # Step 4: Get risk level and recommendations
            risk_level, recommendations = self._get_risk_recommendation(dropout_prob, patient_data)
            
            logger.info(f"Prediction successful: {dropout_prob:.3f} probability, {risk_level} risk")
            logger.info(f"=== PREDICTION COMPLETE ===")
            
            return dropout_prob, confidence_score, risk_level, recommendations
            
        except Exception as e:
            logger.error(f"Error in dropout prediction: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return 0.5, 0.0, "Medium", ["Unable to generate recommendations due to prediction error"]

    def _get_risk_recommendation(self, dropout_prob: float, patient_data: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """ Generating risk level and recommendation"""
        
        if dropout_prob > 0.7:
            return "High", [
                "Schedule immediate follow-up call",
                "Assign dedicated support specialist", 
                "Offer flexible scheduling options",
                "Provide additional motivational support"
            ]
        elif dropout_prob > 0.3:
            return "Medium", [
                "Send weekly check-in messages",
                "Monitor attendance closely",
                "Offer additional support if needed",
                "Track progress more frequently"
            ]
        else:
            return "Low", [
                "Continue standard care procedures",
                "Maintain regular check-ins",
                "Monitor for any changes in engagement"
            ]