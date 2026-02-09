# -*- coding: utf-8 -*-
"""
Gets the data ready for training: we clean it and turn text columns into numbers.
"""
import pandas as pd
from pathlib import Path


def load(csv_path=None):
    if csv_path is None:
        base = Path(__file__).resolve().parent.parent
        csv_path = base / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(csv_path)
    return df


def clean(df):
    df = df.dropna(how="any")
    # TotalCharges is often stored as text in the Telco CSV
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def add_features(df):
    """Derived signals that often help churn models on this dataset."""
    df = df.copy()
    tenure = df["tenure"].replace(0, 1)
    df["avg_monthly_spend"] = df["TotalCharges"] / tenure
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["high_monthly"] = (
        df["MonthlyCharges"] > df["MonthlyCharges"].median()
    ).astype(int)
    return df


def to_numeric(df):
    churn_map = {"No": 0, "Yes": 1}
    if "Churn" in df.columns:
        df = df.copy()
        df["Churn"] = df["Churn"].map(churn_map)
    return df


def prepare_for_model(df, target="Churn"):
    df = add_features(df)
    drop_cols = ["customerID", target]
    cols = [c for c in df.columns if c not in drop_cols]
    X = df[cols].copy()
    y = df[target] if target in df.columns else None

    X = pd.get_dummies(X, drop_first=True)
    return X, y
