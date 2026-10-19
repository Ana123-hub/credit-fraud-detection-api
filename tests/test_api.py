import os
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import app


@pytest.fixture
def valid_transaction_payload():
    return {
        "Time": 86400.0,
        "Amount": 149.99,
        **{f"V{i}": 0.0 for i in range(1, 29)},
    }

class TestFastAPIEndpoints:

    def test_health_check_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_predict_endpoint_success(self, valid_transaction_payload):
        with TestClient(app) as client:
            response = client.post("/predict", json=valid_transaction_payload)
            assert response.status_code == 200
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "status" in data

    def test_predict_rejects_missing_feature(self, valid_transaction_payload):
        payload = valid_transaction_payload.copy()
        payload.pop("V28")

        with TestClient(app) as client:
            response = client.post("/predict", json=payload)

        assert response.status_code == 422

    def test_readiness_check_endpoint(self):
        with TestClient(app) as client:
            response = client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_readiness_fails_when_predictor_unavailable(self, monkeypatch):
        with TestClient(app) as client:
            monkeypatch.setattr(app.state, "predictor", None)
            response = client.get("/ready")

        assert response.status_code == 503
