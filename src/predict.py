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

    def __init__(self, model_path: str = None):
        # Resolve model bundle paths from config if not supplied
        if model_path is None:
            model_path = os.path.join(
                CONFIG["paths"]["models_dir"], CONFIG["paths"]["model_filename"]
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model bundle not found at: {model_path}"
            )
        model_bundle = joblib.load(model_path)

        if not isinstance(model_bundle, dict):
            raise ValueError("Expected a model bundle containing the model, scaler, "
                             "decision threshold and feature names. Retrain the model."
            )

        required_keys = {
            "model",
            "scaler",
            "optimal_threshold",
            "feature_names",
        }

        missing_keys = required_keys - model_bundle.keys()
        if missing_keys:
            raise ValueError(f"Model bundle is missing required keys: "
                             f"{sorted(missing_keys)}. Retrain the model."
            )

        self.model = model_bundle["model"]
        self.scaler = model_bundle["scaler"]
        self.decision_threshold = float(model_bundle["optimal_threshold"])

        self.feature_names = list(model_bundle["feature_names"])
        if not 0 <= self.decision_threshold <= 1:
            raise ValueError("Decision threshold must be between 0 and 1.")

        # Ensure recorded feature order matches the fitted model.
        if hasattr(self.model, "feature_names_in_"):
            model_features = list(self.model.feature_names_in_)

            if model_features != self.feature_names:
                raise ValueError(
                "Bundle feature names do not match the model's expected feature names and order."
                )

    def preprocess_payload(self, raw_dict: Dict[str, Any]) -> pd.DataFrame:
        required_features = (
            {"Time", "Amount"}
            | {f"V{i}" for i in range(1, 29)}
        )
        missing_features = required_features - raw_dict.keys()
        if missing_features:
            raise ValueError("Transaction payload is missing required features: "
                             f"{sorted(missing_features)}"
            )
        df = pd.DataFrame([raw_dict])

        # Cyclical Time encoding
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