"""FastAPI service for a saved H2O AutoML model."""

from contextlib import asynccontextmanager
import json
import math
from pathlib import Path

import h2o
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH_FILE = MODEL_DIR / "model_path.txt"
METADATA_FILE = MODEL_DIR / "metadata.json"

model = None
metadata = {}
feature_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start H2O and load the model once at API startup."""
    global model, metadata, feature_names

    if not MODEL_PATH_FILE.exists() or not METADATA_FILE.exists():
        raise RuntimeError("Model files missing. Run `python train_model.py` first.")

    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    feature_names = metadata["features"]
    model_path = MODEL_PATH_FILE.read_text(encoding="utf-8").strip()
    # A saved path may be absolute on the training machine. In Docker, fall back
    # to the model artifact's filename inside the copied models/ directory.
    if not Path(model_path).exists():
        model_path = str(MODEL_DIR / Path(model_path).name)

    h2o.init()
    try:
        model = h2o.load_model(model_path)
        yield
    finally:
        try:
            h2o.cluster().shutdown(prompt=False)
        except Exception:
            pass


app = FastAPI(
    title="Cloud-Based H2O AutoML Prediction API",
    description="Demo REST API serving a saved H2O AutoML classification model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(
        ..., description="Every feature from GET /features mapped to a number."
    )


@app.get("/")
def root():
    return {"service": "H2O AutoML API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "not_ready",
        "model_loaded": model is not None,
    }


@app.get("/features")
def get_features():
    return {"features": feature_names}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    supplied = request.features
    expected, received = set(feature_names), set(supplied)
    missing, unexpected = sorted(expected - received), sorted(received - expected)
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )

    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"Feature '{name}' must be a finite number.")

    frame = pd.DataFrame([[supplied[name] for name in feature_names]], columns=feature_names)
    try:
        result = model.predict(h2o.H2OFrame(frame)).as_data_frame()
        response = {"prediction": str(result.iloc[0]["predict"])}
        probability_cols = [
            c for c in result.columns if c.startswith("p") and c[1:].isdigit()
        ]
        labels = metadata.get("class_labels", [])
        if probability_cols:
            response["probabilities"] = {
                labels[i] if i < len(labels) else probability_cols[i]:
                    float(result.iloc[0][col])
                for i, col in enumerate(probability_cols)
            }
        return response
    except Exception as exc:
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
