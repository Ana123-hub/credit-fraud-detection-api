# Credit Card Fraud Detection API

An end-to-end machine-learning project for credit-card fraud classification. It includes a configurable preprocessing and training pipeline, a Random Forest model with an F2-optimized decision threshold, a validated FastAPI inference service, automated tests, and a multi-stage Docker image.

The project focuses on the engineering surrounding an ML model: reproducible configuration, train-only preprocessing, a single model bundle, input validation, health/readiness checks, and containerized inference.

## What it does

- Removes duplicate rows and performs a stratified train/test split.
- Encodes the transaction `Time` feature cyclically with sine and cosine features.
- Fits a `RobustScaler` to the training split's `Amount` values and applies that fitted scaler to the test split.
- Trains a Random Forest classifier using `class_weight="balanced_subsample"` to account for class imbalance without synthetic oversampling.
- Uses out-of-fold probabilities from stratified cross-validation to select a decision threshold by F2 score, which weights recall more heavily than precision.
- Saves the estimator, fitted scaler, threshold, expected feature names, holdout PR-AUC, and configuration snapshot in a single model bundle.
- Serves predictions through FastAPI and packages the inference service in Docker.

## Project structure

```text
Credit_Fraud/
├── config.yaml                 # Data, model and path configuration
├── requirements-runtime.txt    # Pinned runtime dependencies
├── pyproject.toml              # Package metadata and development/test setup
├── Dockerfile                  # Multi-stage image build and health check
├── .dockerignore
├── data/
│   ├── creditcard.csv          # Source dataset (provided separately)
│   └── processed/              # Generated train/test data and scaler
├── models/
│   └── random_forest_f2.joblib # Trained model bundle used by inference
├── src/
│   ├── config_loader.py        # Loads and resolves project configuration
│   ├── data_prep.py            # Cleans, splits and transforms the dataset
│   ├── train.py                # Cross-validation, threshold selection and training
│   ├── predict.py              # Bundle validation, preprocessing and inference
│   └── main.py                 # FastAPI application and HTTP endpoints
└── tests/
    ├── test_api.py
    ├── test_data_prep.py
    └── test_predict.py
```

Generated data under `data/` is not required inside the inference container. The container uses the saved bundle in `models/` and the application configuration.

## Dataset

The pipeline expects a CSV with the target column `Class`, transaction time `Time`, amount `Amount`, and anonymized features `V1` through `V28`. `Class` is the label (`0` for legitimate and `1` for fraud) and is removed from the model input features.

Place the dataset at the path configured by `paths.raw_data` in `config.yaml`. The default project path is `data/creditcard.csv`. The raw dataset is not bundled into the Docker image; obtain it separately and comply with the dataset provider's terms.

## Environment setup

The project is configured for Python `>=3.14,<3.15`. Use a compatible Python 3.14 installation.

### Windows PowerShell

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks virtual-environment activation, follow your local Python/PowerShell policy or run the environment's Python executable directly. Do not install project dependencies into a different interpreter from the one used to run tests.

The Docker image installs the pinned packages from `requirements-runtime.txt` and installs the project wheel built from `pyproject.toml`.

## Configuration

`config.yaml` centralizes paths and training parameters. The path settings follow this contract:

- `raw_data`: path to the source CSV.
- `processed_dir`: directory for generated train/test CSVs and the preprocessing scaler artifact.
- `models_dir`: directory for the trained model bundle.
- `model_filename`: filename of the saved model bundle, for example `random_forest_f2.joblib`.
- `scaler_filename`: filename of the preprocessing scaler artifact, for example `robust_scaler.joblib`.

Directory paths and filenames have separate roles: `model_filename` and `scaler_filename` are filenames, not full paths. Relative directory paths are resolved from the project root. In Docker, `CREDIT_FRAUD_PROJECT_ROOT` is set to `/app`.

Training parameters such as the train/test split size, random seed, Random Forest settings and class weighting are also configured in `config.yaml`. Review that file before running a new experiment.

## Prepare data and train the model

Run these commands from the repository root, with the development environment activated and the dataset in the configured location.

### 1. Prepare the data

```powershell
python -m src.data_prep
```

The preparation pipeline:

1. Loads the configured CSV and removes duplicate rows.
2. Separates `Class` from the input features.
3. Makes a stratified train/test split using the configured split size and random seed.
4. Creates `hour_sin` and `hour_cos` by mapping the dataset's elapsed-seconds `Time` feature onto an assumed 86,400-second cycle, then removes the raw `Time` column.
5. Fits `RobustScaler` on the training split's `Amount` column only, transforms the test split with that same fitted scaler, and names the scaled feature `scaled_amount`.
6. Writes `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and the fitted scaler to the configured processed directory.

### 2. Train and evaluate

```powershell
python -m src.train
```

Training uses stratified cross-validation to generate out-of-fold probabilities and select an F2 threshold. The final Random Forest is then fitted on the training split and evaluated on the held-out test split using PR-AUC and classification metrics. The resulting bundle is saved to the configured models directory.

The inference service expects the model bundle to contain the estimator, fitted `RobustScaler`, decision threshold, and expected feature names. It validates this schema when initializing instead of silently accepting an incomplete or incompatible bundle.

## Run the API locally

After the model bundle has been created at the configured location:

```powershell
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

FastAPI interactive documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

### Health and readiness endpoints

| Method and path | Purpose | Expected behavior |
|---|---|---|
| `GET /health` | Liveness: confirms the API process can respond | HTTP `200`, with `status: "ok"` |
| `GET /ready` | Readiness: confirms the predictor is available | HTTP `200`, with `status: "ready"`; HTTP `503` if the predictor is unavailable |
| `POST /predict` | Scores a transaction | Returns the prediction fields or a request-validation error |

The Docker health check calls `/ready`, so a live process that cannot provide predictions is not marked ready.

### Prediction request

The request must include `Time`, `Amount`, and every feature from `V1` to `V28`. Missing fields, extra fields, and non-finite numeric values are rejected by request validation.

Example PowerShell request using synthetic input (all PCA features set to zero):

```powershell
$transaction = @{ Time = 86400.0; Amount = 149.99 }
1..28 | ForEach-Object { $transaction["V$_"] = 0.0 }
$body = $transaction | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

A successful response has this shape:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0002,
  "decision_threshold": 0.3222897592150029,
  "status": "APPROVED_LEGIT"
}
```

The numeric values above illustrate a previously verified synthetic smoke test; predictions and the selected threshold depend on the model bundle in use. They are not a performance claim. The `status` field is `FLAGGED_FRAUD` when the estimated fraud probability meets or exceeds the bundle's threshold, otherwise `APPROVED_LEGIT`.

## Run with Docker

Build the image from the repository root:

```powershell
docker build -t credit-fraud-api:dev .
```

Start the service on port 8000:

```powershell
docker run --rm --name credit-fraud-api -p 8000:8000 credit-fraud-api:dev
```

If port 8000 is already in use on the host, map a different host port, for example `-p 8001:8000`, and use that host port in the URLs below.

Verify readiness and the Docker health status from another PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/ready
docker inspect --format='{{.State.Health.Status}}' credit-fraud-api
```

The readiness response should report `ready`, and Docker's health status should become `healthy` after a successful health-check interval. The image uses a multi-stage build, a non-root runtime user, pinned runtime dependencies, and one Uvicorn worker. The trained model bundle and `config.yaml` are copied into the runtime image; the raw dataset is not needed for inference.

To stop the foreground container, press `Ctrl+C`. To stop a detached container, run `docker stop <container-name>`.

## Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

The tests cover data preparation, preprocessing and inference behavior, API request validation, liveness/readiness behavior, and rejection of an incomplete model bundle. Rerun the suite after code, configuration, dependency, or model-bundle changes.

## Modeling and engineering decisions

- **Stratified split:** keeps the rare fraud class represented in both training and holdout data.
- **Train-only scaling:** the `RobustScaler` is fitted only on training `Amount` values to avoid leaking holdout information into preprocessing.
- **Cyclical time features:** sine/cosine encoding represents proximity around the 24-hour boundary more naturally than a raw linear time value.
- **Class weighting rather than synthetic oversampling:** the Random Forest uses `balanced_subsample`. This avoids generating synthetic records, but it is a modeling choice to validate against alternative imbalance strategies during future experiments.
- **F2 threshold selection:** F2 places greater emphasis on recall than precision, reflecting a preference to reduce missed fraud cases while still tracking false alarms. It does not remove the need to examine the operational cost of false positives.
- **PR-AUC for evaluation:** precision-recall evaluation is informative for a highly imbalanced target. Threshold-dependent precision, recall, confusion matrices and false-positive counts should also be reviewed.
- **Single model bundle:** the estimator, fitted scaler, feature schema and threshold travel together, reducing the risk that inference loads mismatched artifacts.

Performance values depend on the exact dataset, training configuration, dependency versions and saved model artifact. No fixed benchmark figures are reported here because they can become stale when any of those inputs change. Use the latest training output and the saved bundle's `holdout_pr_auc` when publishing results, and report it alongside threshold-dependent metrics such as precision, recall and confusion-matrix counts.

## Limitations and responsible use

- This is a portfolio/demo system using historical transaction data and anonymized PCA features, not a connection to a live card network or bank.
- A successful API response or a healthy Docker container verifies the service path, not the model's production suitability.
- The output is a model estimate and threshold-based label. A real payment decision should also consider business policies, customer impact, monitoring, drift, data quality, and human escalation where appropriate.
- Re-evaluate the model on representative, recent data before any real-world use. Track precision, recall, PR-AUC, false-positive burden, missed fraud, calibration and subgroup behavior where relevant.
- Treat serialized `joblib` model files as trusted artifacts; do not load model files from untrusted sources.
- The dataset does not provide actual transaction timestamps. The periodic features derived from `Time` therefore do not recover the true hour of day and the usefulness of this encoding should be validated on representative data.

## License and dataset

This project uses the [Credit Card Fraud Detection dataset hosted on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Download the dataset separately and place it at the path configured by `paths.raw_data`. The raw CSV is not included in the repository or Docker image.

Review the dataset provider's terms before reuse or redistribution. A project license has not yet been specified in this repository; add an appropriate `LICENSE` file before granting reuse rights to the project code.
