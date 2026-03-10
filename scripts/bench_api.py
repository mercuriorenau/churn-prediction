# -*- coding: utf-8 -*-
"""
Benchmark FastAPI /predict latency via TestClient (no live server required).

Usage from project root (after training so models/ exists):
    python scripts/bench_api.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

import api as api_module


def main(n_warmup=10, n_runs=100):
    model_path = ROOT / "models" / "modelo_churn.pkl"
    cols_path = ROOT / "models" / "columnas.pkl"
    if not model_path.exists() or not cols_path.exists():
        print("Missing models/*.pkl. Run: python src/train.py")
        sys.exit(1)

    api_module.MODEL_PATH = model_path
    api_module.COLS_PATH = cols_path
    api_module.THRESHOLD_PATH = ROOT / "models" / "threshold.pkl"
    if not api_module.load_artifacts():
        print("Failed to load model artifacts.")
        sys.exit(1)

    with open(cols_path, "rb") as f:
        cols = pickle.load(f)
    features = {c: 0.0 for c in cols}
    # Mildly informative defaults for a few known columns
    for key, val in (
        ("tenure", 12.0),
        ("MonthlyCharges", 70.0),
        ("TotalCharges", 840.0),
        ("avg_monthly_spend", 70.0),
        ("is_new_customer", 1.0),
        ("high_monthly", 1.0),
    ):
        if key in features:
            features[key] = val

    client = TestClient(api_module.app)
    payload = {"features": features}

    for _ in range(n_warmup):
        r = client.post("/predict", json=payload)
        r.raise_for_status()

    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        t1 = time.perf_counter()
        r.raise_for_status()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    p95_ms = float(np.percentile(times_ms, 95))
    out = ROOT / "reports" / "api_latency.txt"
    out.parent.mkdir(exist_ok=True)
    text = (
        f"endpoint: POST /predict (FastAPI TestClient)\n"
        f"runs: {n_runs} (warmup: {n_warmup})\n"
        f"mean_ms: {mean_ms:.2f}\n"
        f"p95_ms: {p95_ms:.2f}\n"
    )
    out.write_text(text)
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
