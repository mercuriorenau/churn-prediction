# -*- coding: utf-8 -*-
"""
Simple API to get churn prediction. Run with: uvicorn api:app --reload (from src/) or uvicorn src.api:app (from project root).
"""
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "modelo_churn.pkl"
COLS_PATH = BASE / "models" / "columnas.pkl"
THRESHOLD_PATH = BASE / "models" / "threshold.pkl"

app = FastAPI(title="Churn prediction API")
model = None
COLUMNS = None
THRESHOLD = 0.5


def load_artifacts():
    global model, COLUMNS, THRESHOLD
    if MODEL_PATH.exists() and COLS_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(COLS_PATH, "rb") as f:
            COLUMNS = pickle.load(f)
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH, "rb") as f:
                THRESHOLD = pickle.load(f)
    return model is not None


@app.on_event("startup")
def startup():
    load_artifacts()
    if model is None:
        print("Warning: No model found. Run train.py first. API will return 503 on /predict.")


class PredictRequest(BaseModel):
    features: dict


@app.get("/")
def root():
    return {"message": "Churn prediction API. POST /predict with body {\"features\": {\"col1\": val1, ...}}"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or COLUMNS is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    row = np.array([float(req.features.get(c, 0)) for c in COLUMNS]).reshape(1, -1)
    prob = float(model.predict_proba(row)[0, 1])
    pred = 1 if prob >= THRESHOLD else 0
    return {"churn": pred, "churn_probability": round(prob, 4)}
