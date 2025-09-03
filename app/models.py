# defining the schemas of the fastapi
from pydantic import BaseModel
from typing import List , Dict, Optional

class Patient(BaseModel):    # input
    """patient data schema"""
    patient_id: int
    age: float 
    gender : str
    bmi: float
    smoker : str
    n_sessions : int
    injury_type : str

    class Config:
        orm_mode = True


class DropoutPrediction(BaseModel):   # output
    patient_id : str
    dropout_probability : float
    risk_level : str
    recommendation : List[str]