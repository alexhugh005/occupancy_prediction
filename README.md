# Room Occupancy Prediction – AI Engineering Operationalization

Case study implementation for operationalizing an XGBoost occupancy classifier.

## Dataset

- **Source**: UCI Occupancy Detection dataset (sensor readings + ground-truth occupancy)
- **Features**: Temperature (°C), Humidity (%), Light (Lux), CO2 (ppm), HumidityRatio, datetime
- **Target**: Occupancy (0 = empty, 1 = occupied)
- Files: `data/datatraining.txt`, `data/datatest.txt`, `data/datatest2.txt`

---
# Environment Setup

"python -m pip install -r requirements.txt"

## Task 1: Training Pipeline – Workflow Steps

```
┌─────────────────────┐
│  1. Data Ingestion  │  Load CSV (train + optional test)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Data Validation │  Schema checks, type casting, missing-value handling
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. Feature Eng.    │  Parse datetime → hour, dayofweek, time_of_day, ...
│                     │  One-hot encode categorical datetime features
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. Train / Val     │  Stratified split (or external test set)
│     Split           │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. HPO             │  RandomizedSearchCV over XGBoost hyper-parameters
│     (Random Search) │  Scoring = ROC-AUC
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  6. Final Training  │  Refit best estimator on full training data
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  7. Evaluation      │  ROC-AUC + classification report / confusion matrix
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  8. Artifact Store  │  Persist model.joblib + feature_engineer.joblib
│                     │  + metrics.json (versioned by timestamp)
└─────────────────────┘
```

### Implementation

```bash
# From project root
PYTHONPATH=src python src/train.py \
  --train-path data/datatraining.txt \
  --test-path data/datatest.txt \
  --model-dir models \
  --results-dir results \
  --n-iter 10
```

Key modules:
- `src/feature_engineering.py` – datetime feature extraction + OneHotEncoder
- `src/train.py` – full pipeline orchestration

---

## Task 2: Inference – Event-driven Model Endpoint

A FastAPI service that accepts individual sensor events (or batches) and returns occupancy prediction + probability.

### Endpoints

| Method | Path            | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | /health         | Liveness / readiness                 |
| POST   | /predict        | Single event prediction              |
| POST   | /predict/batch  | Batch of events                      |
| GET    | /docs           | OpenAPI interactive docs             |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2015-02-04 17:51:00",
    "Temperature": 23.18,
    "Humidity": 27.272,
    "Light": 426.0,
    "CO2": 721.25,
    "HumidityRatio": 0.004793
  }'
```

### Local run

```bash
PYTHONPATH=src MODEL_DIR=models uvicorn inference:app --app-dir src --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The service loads `model.joblib` + `feature_engineer.joblib` at startup (or on first request if configured).  
In production the model artifacts would typically be pulled from a model registry (MLflow, S3, etc.) rather than baked into the image.

---

## Task 3: CI/CD / Continuous Training

### CI/CD Pipeline (see `.github/workflows/ci-cd.yml`)

1. **Test job**
   - Install dependencies
   - Run unit tests (`tests/`)
   - Smoke-run the training pipeline with a small search budget

2. **Build & push training container**
   - `docker/Dockerfile.train`
   - Image contains code + data + training entrypoint
   - Pushed to container registry (e.g. GHCR) tagged with git SHA + `latest`

3. **Build & push inference container**
   - `docker/Dockerfile.inference`
   - Contains serving code + (optionally) the latest model artifacts
   - Pushed to the same registry

4. **Smoke-test inference image** (optional integration step)

### Continuous Training (CT) – Verbal Description

Continuous Training keeps the production model fresh as new labeled data arrives or as data/concept drift is detected.

Recommended approach:

1. **Trigger sources**
   - Scheduled (e.g. nightly / weekly via cron or GitHub Actions `schedule`)
   - Event-driven: new labeled batch lands in a data lake / feature store
   - Drift detection: monitoring service (Evidently, custom PSI/KS tests on Light/CO2/Temperature distributions) emits an alert → triggers retrain

2. **Pipeline**
   - Re-use the same training container (`occupancy-train`) 
   - Pull latest training window (or full history + new data)
   - Execute `train.py` → produce new model version + metrics
   - Compare new ROC-AUC (and business metrics) against current production baseline
   - If improved (or within tolerance and drift was high) → promote model

3. **Promotion / Model Registry**
   - Register model in MLflow / Vertex AI / SageMaker Model Registry with:
     - metrics, data hash, code commit SHA, feature-engineer artifact
   - Automated approval gate or human-in-the-loop for production promotion
   - Inference service pulls the new model version (sidecar / init-container / hot-reload) without full redeploy when possible

4. **Feedback loop**
   - Log predictions + ground-truth (when available later) for future training sets
   - Maintain a champion / challenger evaluation on a live shadow traffic slice

This closes the loop: Data → Train → Evaluate → Deploy → Monitor → (re)Train.

---

## Project Layout

```
occupancy_prediction/
├── data/                     # Raw CSVs
├── src/
│   ├── feature_engineering.py
│   ├── train.py
│   └── inference.py
├── models/                   # Trained artifacts (model.joblib, …)
├── tests/
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.inference
├── ci/
│   └── github-actions.yml      # same pipeline, for reference
├── .github/workflows/
│   └── ci-cd.yml               # workflow GitHub Actions runs
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
PYTHONPATH=src python src/train.py

# Serve
PYTHONPATH=src uvicorn inference:app --app-dir src --port 8000

# Docker (training)
docker build -f docker/Dockerfile.train -t occupancy-train .
docker run --rm -v $(pwd)/models:/app/models occupancy-train

# Docker (inference)
docker build -f docker/Dockerfile.inference -t occupancy-inference .
docker run --rm -p 8000:8000 occupancy-inference
```
