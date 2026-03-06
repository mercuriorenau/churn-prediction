# -*- coding: utf-8 -*-
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import api as api_module


class FakeModel:
    def predict_proba(self, X):
        n = len(X)
        # Always return high churn probability for predictable assertions
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


@pytest.fixture
def artifact_dir(tmp_path):
    cols = ["tenure", "MonthlyCharges", "avg_monthly_spend"]
    model_path = tmp_path / "modelo_churn.pkl"
    cols_path = tmp_path / "columnas.pkl"
    thr_path = tmp_path / "threshold.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(FakeModel(), f)
    with open(cols_path, "wb") as f:
        pickle.dump(cols, f)
    with open(thr_path, "wb") as f:
        pickle.dump(0.5, f)
    return tmp_path, cols


@pytest.fixture
def client_with_model(artifact_dir, monkeypatch):
    tmp_path, cols = artifact_dir
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "modelo_churn.pkl")
    monkeypatch.setattr(api_module, "COLS_PATH", tmp_path / "columnas.pkl")
    monkeypatch.setattr(api_module, "THRESHOLD_PATH", tmp_path / "threshold.pkl")
    monkeypatch.setattr(api_module, "model", None)
    monkeypatch.setattr(api_module, "COLUMNS", None)
    monkeypatch.setattr(api_module, "THRESHOLD", 0.5)
    assert api_module.load_artifacts() is True
    return TestClient(api_module.app), cols


def test_root_ok(client_with_model):
    client, _ = client_with_model
    r = client.get("/")
    assert r.status_code == 200
    assert "predict" in r.json()["message"].lower()


def test_predict_returns_churn_and_probability(client_with_model):
    client, cols = client_with_model
    features = {c: 1.0 for c in cols}
    r = client.post("/predict", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert "churn" in body
    assert "churn_probability" in body
    assert body["churn"] in (0, 1)
    assert body["churn"] == 1
    assert body["churn_probability"] == 0.8


def test_predict_503_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(api_module, "COLS_PATH", tmp_path / "missing_cols.pkl")
    monkeypatch.setattr(api_module, "THRESHOLD_PATH", tmp_path / "missing_thr.pkl")
    monkeypatch.setattr(api_module, "model", None)
    monkeypatch.setattr(api_module, "COLUMNS", None)
    with TestClient(api_module.app) as client:
        r = client.post("/predict", json={"features": {"tenure": 1}})
        assert r.status_code == 503
