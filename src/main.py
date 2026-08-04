import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import FraudPredictor

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time transaction fraud probability scoring service",
    version="1.0.0"
)

# Global predictor instance initialized on startup
predictor_instance = None

@app.on_event("startup")
def load_model():
    global predictor_instance
    predictor_instance = FraudPredictor()

@app.get("/health")
def health_check():
    """Health check endpoint for container monitoring and readiness tests."""
    return {"status": "ok", "service": "fraud-detection-api"}

class TransactionPayload(BaseModel):
    Time: float = Field(..., json_schema_extra={"example": 86400.0})
    Amount: float = Field(..., json_schema_extra={"example": 149.99})
    # Accept V1 through V28 dynamically as a dict or allow extra fields
    class Config:
        extra = "allow"

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    if predictor_instance is None:
        raise HTTPException(status_code=500, detail="Model predictor is not initialized.")
    try:
        return predictor_instance.predict(payload)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))