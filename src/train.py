import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve,
    average_precision_score
)

# Add parent directory (Credit_Fraud) to Python sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Centralized Configuration Import
from src.config_loader import CONFIG


def train_model():
    """
    Loads preprocessed datasets from configured directories, tunes the decision threshold 
    via 5-Fold Stratified CV, fits the final Random Forest model using configured hyperparameters, 
    and saves the model bundle to the target models directory.
    """
    # Resolve Paths & Configuration Options
    processed_dir = CONFIG["paths"]["processed_dir"]
    models_dir = CONFIG["paths"]["models_dir"]
    model_save_path = os.path.join(CONFIG["paths"]["models_dir"], CONFIG["paths"]["model_filename"])
    # Make sure directory exists before saving
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    rf_params = CONFIG["model_params"]

    print(f" Loading Processed Data From: {processed_dir}")
    print(f" Saving Trained Model To:     {model_save_path}\n")

    # Load Preprocessed Data
    print("--- Step 1: Loading preprocessed data ---")
    try:
        X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
        X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Failed to find preprocessed data in '{processed_dir}'. "
        ) from e

    print(f"X_train shape: {X_train.shape} | Fraud cases: {y_train.sum()}")
    print(f"X_test shape:  {X_test.shape}  | Fraud cases: {y_test.sum()}")

    # 5-Fold Stratified Cross-Validation
    print("\n--- Step 2: Running 5-Fold CV for Out-Of-Fold F2 Threshold Optimization ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rf_params["random_state"])
    oof_probs = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        rf_fold = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            max_features='sqrt',
            class_weight=rf_params["class_weight"],
            n_jobs=-1,
            random_state=rf_params["random_state"]
        )
        rf_fold.fit(X_tr, y_tr)
        oof_probs[val_idx] = rf_fold.predict_proba(X_va)[:, 1]

    # Calculate optimal F2 threshold on Out-Of-Fold (OOF) probabilities
    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_f2_idx = np.argmax(f2_scores)
    optimal_threshold = float(thresholds[best_f2_idx])

    print(f" Optimal F2 Threshold: {optimal_threshold:.4f}")
    print(f" OOF Precision at Threshold: {precisions[best_f2_idx]:.4f}")
    print(f" OOF Recall at Threshold:    {recalls[best_f2_idx]:.4f}")

    # Fit Final Model on 100% Training Data
    print("\n--- Step 3: Fitting Final Random Forest Model ---")
    final_rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        max_features='sqrt',
        class_weight=rf_params["class_weight"],
        n_jobs=-1,
        random_state=rf_params["random_state"]
    )
    final_rf.fit(X_train, y_train)

    # Evaluate on Holdout Test Set
    test_probs = final_rf.predict_proba(X_test)[:, 1]
    holdout_pr_auc = average_precision_score(y_test, test_probs)
    test_preds = (test_probs >= optimal_threshold).astype(int)

    print(f"\n Holdout Test PR-AUC: {holdout_pr_auc:.4f}")
    print("\n--- Holdout Test Classification Report ---")
    print(classification_report(y_test, test_preds, target_names=['Legit (0)', 'Fraud (1)']))

    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}  | True Positives:  {cm[1,1]}")

    # Save Artifact
    print(f"\n--- Step 4: Saving model bundle to '{models_dir}/' ---")
    os.makedirs(models_dir, exist_ok=True)

    model_bundle = {
        "model": final_rf,
        "optimal_threshold": optimal_threshold,
        "feature_names": list(X_train.columns),
        "holdout_pr_auc": holdout_pr_auc,
        "config_snapshot": CONFIG
    }

    joblib.dump(model_bundle, model_save_path)

    print(f" Model bundle successfully saved to: {model_save_path}")
    print(" Model training pipeline complete!")


if __name__ == "__main__":
    train_model()