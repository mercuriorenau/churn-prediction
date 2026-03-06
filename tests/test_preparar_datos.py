# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preparar_datos import add_features, clean, prepare_for_model, to_numeric
from tests.fixtures_data import sample_telco_df


def test_clean_coerces_total_charges():
    df = sample_telco_df()
    out = clean(df)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].isna().sum() == 0


def test_to_numeric_maps_churn():
    df = to_numeric(clean(sample_telco_df()))
    assert set(df["Churn"].unique()).issubset({0, 1})


def test_add_features_creates_derived_columns():
    df = add_features(clean(sample_telco_df()))
    for col in ("avg_monthly_spend", "is_new_customer", "high_monthly"):
        assert col in df.columns


def test_prepare_for_model_returns_matrix():
    df = to_numeric(clean(sample_telco_df()))
    X, y = prepare_for_model(df)
    assert len(X) == len(df)
    assert y is not None
    assert len(y) == len(df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert "avg_monthly_spend" in X.columns
