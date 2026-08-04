import os
import sys
import pytest
from fastapi.testclient import TestClient
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import app


@pytest.fixture
def valid_transaction_payload():
    payload = {
        "Time": 86400.0,
        "Amount": 149.99,
    }
    # Populate V1 through V28
    for i in range(1, 29):
        payload[f"V{i}"] = -0.5
    return payload


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