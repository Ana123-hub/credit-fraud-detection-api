# Credit Card Fraud Detection Pipeline and API

An end-to-end, production-ready machine learning system built with scikit-learn and FastAPI to detect high-risk, fraudulent financial transactions in real time. This project showcases the evolution of a production pipeline, from raw data cleaning and cyclical feature engineering to cost-sensitive threshold optimization, rigorous test suites, and containerized microservice deployment with Docker.

# Architecture and Production Stack

```text
Credit_Fraud/
├── data/
│   ├── raw/                  # Raw creditcard.csv transaction dataset
│   └── processed/            # Serialized feature processing artifacts
├── models/                   # Serialized model artifacts & scalers
│   └── random_forest_f2.joblib
├── src/
│   ├── main.py               # FastAPI application server & endpoint handlers
│   ├── predict.py            # Feature transformation & inference pipeline
│   ├── data_prep.py          # Data cleaning, scaling & feature transformers
│   ├── train.py              # Model training & validation pipeline
│   └── config_loader.py      # Environment & system configurations
├── tests/                    # Automated unit & integration test suite
│   ├── test_api.py           # FastAPI endpoint tests
│   ├── test_predict.py       # Inference pipeline tests
│   └── test_data_prep.py     # Data transformation & feature engineering unit tests
├── Dockerfile                # Multi-stage container blueprint (Python 3.11-slim)
├── docker-compose.yml        # Multi-container orchestration spec
├── config.yaml               # System parameters & path configurations
├── pyproject.toml            # Project dependencies & build metadata
└── README.md
```

* API Engine: FastAPI + Uvicorn
* Testing Suite: Pytest + HTTPX
* Model Runtime: Python 3.11-slim container environment
* ML Stack: Scikit-learn, Pandas, NumPy, Joblib
* Containerization: Docker Desktop + Docker Compose

## Multi-Model Performance Summary

All models were evaluated on a **5-Fold Cross-Validation** setup on $80\%$ training data and validated on a strict **20% Stratified Holdout Test Set** ($56,746$ total transactions: $56,651$ Legit, $95$ Fraud). Models were optimized using **Precision-Recall AUC (PR-AUC)**, **Fraud Recall**, and **Decision Threshold Tuning** ($F_1$ vs. $F_2$).

| Model Architecture | 5-Fold Mean CV PR-AUC | Holdout Test PR-AUC | Optimal Threshold ($\tau$) | Metric Optimized | Precision (Fraud) | Recall (Fraud) | $F_1$ Score | True Positives (Caught) | False Negatives (Missed) | False Positives (Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest ($F_2$-Tuned)** | **0.8311** | **0.7965** | **$\tau = 0.3274$** | **$F_2$ Score** | **0.81** | **0.79** | **0.80** | **75 / 95** | **20** | **18** |
| **Random Forest ($F_1$-Tuned)** | 0.8311 | 0.7965 | $\tau = 0.5444$ | $F_1$ Score | 0.93 | 0.74 | 0.82 | 70 / 95 | 25 | 5 |
| **CatBoost Classifier** | 0.7357 | 0.8019 | $\tau = 0.7814$ | $F_2$ Score | 0.90 | 0.78 | 0.84 | 74 / 95 | 21 | 8 |
| **AdaBoost Classifier** | 0.7296 | 0.7008 | $\tau = 0.8262$ | $F_2$ Score | 0.83 | 0.72 | 0.77 | 68 / 95 | 27 | 14 |
| **LightGBM Classifier** | 0.4473 | 0.6016 | Baseline | Standard | — | — | — | — | — | — |

---

## Detailed Model Breakdown & Execution Logs

### 1. Random Forest Classifier (Selected for Production)
* **Cross-Validation:** **Mean CV PR-AUC: 0.8311** (Folds: `0.8383`, `0.8555`, `0.7638`, `0.8494`, `0.8486`)
* **Holdout Test PR-AUC:** `0.7965`
* **Threshold Tradeoff Analysis:**
  * **$F_1$-Optimized ($\tau = 0.5444$):** High Precision (`0.93`), but missed **25 fraudulent transactions** (Recall: `0.74`).
  * **$F_2$-Optimized ($\tau = 0.3274$):** Lowered threshold to prioritize catching fraud. Reduced False Negatives from **25 down to 20** (Recall: `0.79`), catching **75 out of 95 fraud cases** with only 18 false alarms out of 56,651 legitimate transactions.

```text
--- Random Forest Confusion Matrix (F2 Threshold = 0.3274) ---
True Negatives:  56,633 | False Positives: 18
False Negatives:     20 | True Positives:  75
```
### 2. CatBoost Classifier
* Cross-Validation: Mean CV PR-AUC: 0.7357 (Folds: 0.7891, 0.7523, 0.6566, 0.7113, 0.7694)
* Holdout Test PR-AUC: 0.8019
* Performance at $F_2$ Threshold ($\tau = 0.7814$): Achieved 0.90 Precision and 0.78 Recall ($F_1 = 0.84$), catching 74 out of 95 fraud cases with only 8 false positives.

```text
--- CatBoost Confusion Matrix (F2 Threshold = 0.7814) ---
True Negatives:  56,643 | False Positives: 8
False Negatives:     21 | True Positives:  74
```
### 3. AdaBoost Classifier
* Cross-Validation: Mean CV PR-AUC: 0.7296 (Folds: 0.7284, 0.7506, 0.6897, 0.7229, 0.7565)
* Holdout Test PR-AUC: 0.7008
* Performance at $F_2$ Threshold ($\tau = 0.8262$): Achieved 0.83 Precision and 0.72 Recall ($F_1 = 0.77$), leaving 27 missed frauds (False Negatives).

```text
--- AdaBoost Confusion Matrix (F2 Threshold = 0.8262) ---
True Negatives:  56,637 | False Positives: 14
False Negatives:     27 | True Positives:  68
```
### 4. LightGBM Classifier
* Cross-Validation: Mean CV PR-AUC: 0.4473 (Folds: 0.4246, 0.4364, 0.3149, 0.4701, 0.5903)
* Holdout Test PR-AUC: 0.6016
* Takeaway: LightGBM suffered severe instability across folds without deep hyperparameter tuning, making it unsuitable for deployment compared to Random Forest or CatBoost.


## Technical Decisions & Modeling Rationale

### 1. Exclusion of SMOTE (Synthetic Minority Over-sampling Technique)
While SMOTE is standard for imbalanced classification, it was intentionally omitted in favor of algorithmic cost-sensitive learning (`class_weight="balanced_subsample"`) for three primary production reasons:

* **Inference Data Leakage & Manifold Distortion:** SMOTE creates synthetic instances along linear segments connecting $k$-nearest neighbors. In high-dimensional, PCA-transformed spaces ($V_1$–$V_28$), generating synthetic fraud instances distorts the natural decision boundaries, causing artificial feature overlaps and elevating False Positive rates on unseen live data.
* **Preservation of Real-World Data Distribution:** Modern tree ensembles handle class imbalance natively during node splits by weighing empirical loss based on class frequencies. Retaining the true class ratio ($0.172\%$ fraud) ensures that the probability calibration of the model outputs aligns with live operational distributions.
* **Latency and Memory Efficiency:** Avoiding pre-processing resampling loops simplified the pipeline DAG, eliminating training memory overhead and preventing stateful dependencies during deployment.

---

### 2. Selection of Random Forest for Production Deployment
Despite competing against gradient boosting algorithms (CatBoost, LightGBM, AdaBoost), **Random Forest** was selected as the final production engine due to structural and operational advantages:

* **Robustness to High-Dimensional PCA Noise:** Random Forest applies random feature subspace sampling at every split. In a dataset where features $V_1$ through $V_28$ are orthogonal PCA components, this prevents dominant features from overshadowing subtle signal combinations in lower-variance components.
* **Low Generalization Variance:** As a bagging architecture, Random Forest averages independent decision trees, yielding significantly lower variance and greater resistance to extreme financial outliers than iterative boosting algorithms (e.g., LightGBM, which suffered from severe CV fold variance with a Mean PR-AUC of `0.4473`).
* **Superior Cross-Validation Stability:** Random Forest demonstrated the highest and most consistent cross-validation score (**5-Fold Mean PR-AUC: 0.8311** across folds: `0.8383`, `0.8555`, `0.7638`, `0.8494`, `0.8486`), translating directly to robust performance on the holdout set (`0.7965`).

---

### 3. $F_2$-Score Optimization over Traditional $F_1$-Score Thresholding
Standard binary classifiers default to a $\tau = 0.50$ decision threshold, optimizing for Accuracy or $F_1$-Score. In financial fraud microservices, this introduces a severe misaligned cost structure:

$$\text{Cost of False Negative (Missed Fraud)} \ggg \text{Cost of False Positive (False Alarm)}$$

* **Asymmetric Financial Risk:** A **False Negative** results in unrecovered financial theft, chargebacks, and regulatory risk. A **False Positive** triggers a minor automated flag or step-up authentication.
* **Mathematical Weighting ($F_2$-Metric):** The $F_2$-Score places **twice as much emphasis on Recall as Precision**:
  $$F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{(4 \cdot \text{Precision}) + \text{Recall}}$$
* **Empirical Optimization Impact:** Tuning the decision boundary down to **$\tau = 0.3274$** shifted the performance footprint dramatically:
  * **$F_1$ Threshold ($\tau = 0.5444$):** Precision `0.93`, Recall `0.74` $\rightarrow$ **25 Missed Frauds** (70 Caught).
  * **$F_2$ Threshold ($\tau = 0.3274$):** Precision `0.81`, Recall `0.79` $\rightarrow$ **20 Missed Frauds** (75 Caught).
  * **Result:** Caught **5 additional fraudulent transactions** (+7.1% fraud recovery) on the holdout test set with only 18 false alarms across $56,651$ legitimate transactions ($0.03\%$ false positive rate).

---

### 4. Cyclical Temporal Feature Engineering
Transaction datasets often include raw time deltas (seconds elapsed since the first recorded transaction). Feeding raw linear time into decision trees creates arbitrary splitting thresholds that fail to generalize to future operational days.

* **Sine/Cosine Transformation:** The continuous `Time` feature was transformed into cyclical diurnal features using periodic trigonometric functions ($24$-hour cycle = $86,400$ seconds):
  $$\text{Time}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Time}}{86400}\right), \quad \text{Time}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Time}}{86400}\right)$$
* **Production Rationale:** This maps $23:59$ ($86,340\text{s}$) and $00:01$ ($60\text{s}$) as adjacent points in continuous vector space, allowing the Random Forest model to capture late-night fraud spikes without temporal boundary discontinuities.

## Deployment and Containerization (Docker + FastAPI)
The production pipeline is packaged using multi-stage Docker builds to guarantee runtime parity across environments and keep image footprints minimal.

###1. Build and Launch via Docker Compose
```bash
docker compose up --build -d
```

###2. Monitor Container Status & Health
```bash
docker compose ps
```

## API Inference Quickstart (PowerShell)
Send a transaction payload to receive real-time fraud probability scoring and decision flags:

```text
$body = @{
    Time = 86400.0
    Amount = 149.99
    V1 = -1.359807
    V2 = -0.072781
    V3 = 2.536347
    V4 = 1.378155
    V5 = -0.338321
    V6 = 0.462388
    V7 = 0.239599
    V8 = 0.098698
    V9 = 0.363787
    V10 = 0.090794
    V11 = -0.551600
    V12 = -0.617801
    V13 = -0.991390
    V14 = -0.311169
    V15 = 1.468177
    V16 = -0.470401
    V17 = 0.207971
    V18 = 0.025791
    V19 = 0.403993
    V20 = 0.251412
    V21 = -0.018307
    V22 = 0.277838
    V23 = -0.110474
    V24 = 0.066928
    V25 = 0.128539
    V26 = -0.189115
    V27 = 0.133558
    V28 = -0.021053
}| ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json"
```
## Sample Response:
```text
{
"status": "success",
  "data": {
    "is_fraud": false,
    "fraud_probability": 0.0234,
    "decision_threshold": 0.3200
  }
} 
```