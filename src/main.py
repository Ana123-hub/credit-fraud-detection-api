import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import FraudPredictor

class TransactionPayload(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = FraudPredictor()
    try:
        yield
    finally:
        app.state.predictor = None

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting potentially fraudulent transactions.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health_check():
    """Liveness check: the API process is responding."""
    return {
        "status": "ok",
        "service": "fraud-detection-api",
    }


@app.get("/ready")
def readiness_check(request: Request):
    """Readiness check: the predictor is available for inference."""
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is unavailable.",
        )

    return {
        "status": "ready",
        "service": "fraud-detection-api",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: TransactionPayload,
    request: Request,
):
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is unavailable.",
        )

    return predictor.predict(payload.model_dump())