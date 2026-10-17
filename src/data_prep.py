import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Add parent directory (Credit_Fraud) to Python sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config_loader import CONFIG


def prepare_data():
    """
    Executes a leak-free data preprocessing pipeline using centralized config:
    1. Reads raw data from paths specified in config.yaml.
    2. Performs Stratified Train/Test split according to configured test_size.
    3. Fits RobustScaler exclusively on Train set.
    4. Saves processed datasets and fitted scaler to configured processed directory.
    """
    # Resolve Paths & Config Params
    raw_data_path = CONFIG["paths"]["raw_data"]
    processed_dir = CONFIG["paths"]["processed_dir"]
    scaler_path = os.path.join(processed_dir, CONFIG["paths"]["scaler_filename"])

    test_size = CONFIG["data_prep"]["test_size"]
    random_state = CONFIG["data_prep"]["random_state"]

    print(f" Raw Dataset Path:    {raw_data_path}")
    print(f" Output Directory:    {processed_dir}\n")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Could not find dataset at '{raw_data_path}'. "
            "Ensure 'creditcard.csv' is placed inside 'data/' as defined in config.yaml."
        )

    # Load & Clean Raw Data
    print("--- Step 1: Loading raw data ---")
    df = pd.read_csv(raw_data_path).drop_duplicates().reset_index(drop=True)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    y = df['Class']
    X_raw = df.drop(columns=['Class'])

    print(f"Total rows: {len(df):,} | Fraud count: {y.sum():,} ({y.mean()*100:.4f}%)")

    # Stratified Train/Test Split
    print(f"\n--- Step 2: Splitting Data FIRST ({int((1-test_size)*100)}/{int(test_size*100)} Stratified) ---")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"X_train_raw shape: {X_train_raw.shape} | Train Frauds: {y_train.sum()}")
    print(f"X_test_raw shape:  {X_test_raw.shape}  | Test Frauds:  {y_test.sum()}")

    # Feature Engineering & Scaling
    print("\n--- Step 3: Engineering Features & Fitting Scaler ---")
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Cyclical Time Encoding
    for data in [X_train, X_test]:
        hour = (data['Time'] / 3600) % 24
        data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        data.drop(columns=['Time'], inplace=True)

    # RobustScaler fit STRICTLY on X_train
    scaler = RobustScaler()
    X_train['scaled_amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test['scaled_amount'] = scaler.transform(X_test[['Amount']])

    X_train.drop(columns=['Amount'], inplace=True)
    X_test.drop(columns=['Amount'], inplace=True)

    print(f"Processed feature count: {X_train.shape[1]}")

    # Save preprocessed artifacts
    print(f"\n--- Step 4: Saving preprocessed artifacts to '{processed_dir}/' ---")
    os.makedirs(processed_dir, exist_ok=True)

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    joblib.dump(scaler, scaler_path)

    print(f" Scaler saved: {scaler_path}")
    print(" Data prep execution completed successfully!")


if __name__ == "__main__":
    prepare_data()