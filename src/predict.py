import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config_loader import CONFIG


class FraudPredictor:
    """
    Inference service for the Credit Card Fraud Detection Model.
    Handles raw incoming payloads, feature engineering, scaling, and threshold decision.
    """

    def __init__(self, scaler_path: str = None, model_path: str = None):
        # Resolve artifact paths from config if not provided
        if scaler_path is None:
            scaler_path = os.path.join(
                CONFIG["paths"]["processed_dir"], CONFIG["paths"]["scaler_filename"]
            )
        if model_path is None:
            model_path = os.path.join(
                CONFIG["paths"]["models_dir"], CONFIG["paths"]["model_filename"]
            )

        # Validate existence of model artifacts
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # Load artifacts
        self.scaler = joblib.load(scaler_path)
        
        # Support both raw model objects and dict bundles containing optimal_threshold
        model_bundle = joblib.load(model_path)
        if isinstance(model_bundle, dict):
            self.model = model_bundle.get("model")
            self.decision_threshold = model_bundle.get("optimal_threshold", 0.5)
        else:
            self.model = model_bundle
            self.decision_threshold = getattr(CONFIG, "optimal_threshold", 0.5)

        # Retrieve required model feature names if available, else standard feature list
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            v_cols = [f"V{i}" for i in range(1, 29)]
            self.feature_names = v_cols + ["hour_sin", "hour_cos", "scaled_amount"]

    def preprocess_payload(self, raw_dict: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([raw_dict])

        # Fill missing V1..V28 with default 0.0 if omitted in light payloads
        for i in range(1, 29):
            col = f"V{i}"
            if col not in df.columns:
                df[col] = 0.0

        if "Time" not in df.columns or "Amount" not in df.columns:
            raise KeyError("Payload must include 'Time' and 'Amount' fields.")

        # Cyclical Time Encoding
        hours = (df["Time"] / 3600.0) % 24
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # Scale Amount
        df["scaled_amount"] = self.scaler.transform(df[["Amount"]]).ravel()

        # Drop raw columns
        df = df.drop(columns=["Time", "Amount"])

        return df[self.feature_names]

    def predict(self, raw_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes preprocessing and model inference on a raw transaction dict.
        Returns a dictionary matching the API schema requirements.
        """
        processed_df = self.preprocess_payload(raw_dict)

        # Calculate prediction probability for positive class (Fraud = 1)
        prob = float(self.model.predict_proba(processed_df)[0][1])
        is_fraud = bool(prob >= self.decision_threshold)

        return {
            "is_fraud": is_fraud,
            "fraud_probability": round(prob, 4),
            "decision_threshold": self.decision_threshold,
            "status": "FLAGGED_FRAUD" if is_fraud else "APPROVED_LEGIT"
        }