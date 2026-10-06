"""
Model inference service (event-driven predictions).
Exposes a FastAPI endpoint that accepts sensor readings + timestamp
and returns occupancy prediction (0/1) + probability.
"""

import os
from typing import Optional, List
from datetime import datetime

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from feature_engineering import FeatureEngineer

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    """Single sensor event payload."""
    date: str = Field(..., description="ISO-like datetime string, e.g. '2015-02-04 17:51:00'")
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2015-02-04 17:51:00",
                "Temperature": 23.18,
                "Humidity": 27.272,
                "Light": 426.0,
                "CO2": 721.25,
                "HumidityRatio": 0.004793,
            }
        }


class PredictionResponse(BaseModel):
    occupancy: int = Field(..., description="Predicted class: 0 = not occupied, 1 = occupied")
    probability: float = Field(..., description="Probability of occupancy (class 1)")
    model_version: str = "latest"


class BatchPredictionRequest(BaseModel):
    readings: List[SensorReading]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


# ---------------------------------------------------------------------------
# Application & model loading
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Room Occupancy Prediction API",
    description="Event-driven inference endpoint for XGBoost occupancy classifier",
    version="1.0.0",
)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
_model = None
_fe: Optional[FeatureEngineer] = None


def load_artifacts(model_dir: str = MODEL_DIR) -> None:
    global _model, _fe
    model_path = os.path.join(model_dir, "model.joblib")
    fe_path = os.path.join(model_dir, "feature_engineer.joblib")
    if not os.path.exists(model_path) or not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. Run training first."
        )
    _model = joblib.load(model_path)
    _fe = FeatureEngineer.load(fe_path)
    print(f"Loaded model and feature engineer from {model_dir}")


@app.on_event("startup")
def startup_event():
    try:
        load_artifacts()
    except FileNotFoundError as e:
        print(f"WARNING: {e}. Endpoint will fail until model is available.")
    print("Interactive API docs: http://127.0.0.1:8000/docs")


def _predict_df(df: pd.DataFrame) -> List[PredictionResponse]:
    if _model is None or _fe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = _fe.transform(df)
    proba = _model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return [
        PredictionResponse(occupancy=int(p), probability=float(pr))
        for p, pr in zip(preds, proba)
    ]


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_dir": MODEL_DIR,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """Event-driven single prediction (typical sensor message)."""
    df = pd.DataFrame([reading.model_dump()])
    results = _predict_df(df)
    return results[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictionRequest):
    """Batch prediction for multiple events."""
    records = [r.model_dump() for r in req.readings]
    df = pd.DataFrame(records)
    results = _predict_df(df)
    return BatchPredictionResponse(predictions=results)


@app.get("/")
def root():
    return {
        "service": "Room Occupancy Prediction",
        "endpoints": ["/health", "/predict", "/predict/batch", "/docs"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
