import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config_loader import CONFIG
from src.predict import FraudPredictor


@pytest.fixture(scope="module")
def predictor():
    """Load the predictor using the configured model bundle."""
    model_path = os.path.join(
        CONFIG["paths"]["models_dir"],
        CONFIG["paths"]["model_filename"],
    )

    if not os.path.isfile(model_path):
        pytest.fail(
            f"Model bundle not found at {model_path}. "
            "Run data preparation and training first."
        )

    return FraudPredictor(model_path=model_path)



class TestFraudPredictor:

    def test_predictor_initialization(self, predictor):
        """Verify model, scaler, and decision_threshold are correctly loaded into memory."""
        assert predictor.scaler is not None
        assert predictor.model is not None
        # Checks for 'decision_threshold' or 'optimal_threshold'
        has_thresh = hasattr(predictor, "decision_threshold") or hasattr(predictor, "optimal_threshold")
        assert has_thresh, "FraudPredictor must have a threshold attribute."

    def test_predict_output_schema(self, predictor, valid_transaction_payload):
        """Ensure prediction returns expected keys matching actual FraudPredictor output."""
        result = predictor.predict(valid_transaction_payload)

        assert isinstance(result, dict)
        assert "is_fraud" in result
        assert "fraud_probability" in result
        
        # Matches your actual return payload structure
        assert "decision_threshold" in result or "threshold_used" in result
        assert "status" in result

        assert isinstance(result["is_fraud"], (bool, np.bool_))
        assert isinstance(result["fraud_probability"], float)
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_predict_batch_consistency(self, predictor, valid_transaction_payload):
        """Verify deterministic outputs for identical inputs."""
        res1 = predictor.predict(valid_transaction_payload)
        res2 = predictor.predict(valid_transaction_payload)

        assert res1["fraud_probability"] == res2["fraud_probability"]
        assert res1["is_fraud"] == res2["is_fraud"]

    def test_missing_feature_raises_key_error(self, predictor, valid_transaction_payload):
        """Ensure missing required features raise an error."""
        incomplete_payload = valid_transaction_payload.copy()
        del incomplete_payload["Amount"]

        with pytest.raises((KeyError, ValueError)):
            predictor.predict(incomplete_payload)