import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config_loader import CONFIG
from src.data_prep import prepare_data


@pytest.fixture
def mock_raw_csv(tmp_path):
    """
    Creates a small, temporary mock raw creditcard.csv file 
    to test data_prep without reading the heavy production dataset.
    """
    # Create 20 sample rows matching raw creditcard.csv structure
    data = {
        "Time": np.linspace(0, 100000, 20),
        "Amount": np.random.uniform(10, 500, 20),
        "Class": [0] * 18 + [1] * 2  # Imbalanced class target
    }
    # Add V1 through V28 columns
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(20)

    df = pd.DataFrame(data)
    mock_csv_path = tmp_path / "mock_creditcard.csv"
    df.to_csv(mock_csv_path, index=False)
    return str(mock_csv_path)


class TestDataPrepPipeline:

    def test_raw_data_file_exists(self):
        """Check if raw creditcard.csv exists in configured location."""
        raw_path = CONFIG["paths"]["raw_data"]
        assert os.path.exists(raw_path), (
            f"Raw dataset not found at {raw_path}. "
            "Ensure creditcard.csv is in data/"
        )

    def test_prepare_data_output_artifacts(self, monkeypatch, tmp_path):
        """
        Runs prepare_data() using a temporary directory to verify that all
        processed files (X_train, X_test, y_train, y_test, scaler) are generated.
        """
        # Override processed_dir path in CONFIG to point to tmp_path
        temp_processed_dir = str(tmp_path / "processed")
        monkeypatch.setitem(CONFIG["paths"], "processed_dir", temp_processed_dir)

        # Run pipeline
        prepare_data()

        # Verify output files exist
        assert os.path.exists(os.path.join(temp_processed_dir, "X_train.csv"))
        assert os.path.exists(os.path.join(temp_processed_dir, "X_test.csv"))
        assert os.path.exists(os.path.join(temp_processed_dir, "y_train.csv"))
        assert os.path.exists(os.path.join(temp_processed_dir, "y_test.csv"))
        assert os.path.exists(
            os.path.join(temp_processed_dir, CONFIG["paths"]["scaler_filename"])
        )

    def test_feature_engineering_transformations(self, tmp_path, monkeypatch):
        """
        Verifies that raw features (Time, Amount) are correctly transformed 
        into engineered features (hour_sin, hour_cos, scaled_amount).
        """
        temp_processed_dir = str(tmp_path / "processed")
        monkeypatch.setitem(CONFIG["paths"], "processed_dir", temp_processed_dir)

        prepare_data()

        X_train = pd.read_csv(os.path.join(temp_processed_dir, "X_train.csv"))

        # Verify original raw features were dropped
        assert "Time" not in X_train.columns
        assert "Amount" not in X_train.columns

        # Verify engineered features exist
        assert "hour_sin" in X_train.columns
        assert "hour_cos" in X_train.columns
        assert "scaled_amount" in X_train.columns

        # Verify cyclical sine/cosine bounds [-1.0, 1.0]
        assert X_train["hour_sin"].between(-1.0, 1.0).all()
        assert X_train["hour_cos"].between(-1.0, 1.0).all()