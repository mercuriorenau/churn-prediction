# -*- coding: utf-8 -*-
"""
Interactive dashboard for churn:
- Overview: churn rate and quick charts.
- Segment explorer: filter by contract, tenure and monthly charges.
- Quick scoring: simple form to get churn risk for a synthetic customer.

Run from the project root with:
    streamlit run src/dashboard.py
"""
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from preparar_datos import clean, to_numeric, prepare_for_model


base = Path(__file__).resolve().parent.parent
csv_path = base / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not csv_path.exists():
    st.warning("Put the Telco CSV in the data/ folder to view the dashboard.")
    st.stop()

df_raw = pd.read_csv(csv_path)
df_raw = df_raw.dropna(how="any")

st.title("Churn dashboard")
st.write(
    "Quick view of churn, simple segment filters, and a tiny form to get a churn score "
    "for a synthetic customer."
)

tab_overview, tab_segment, tab_score = st.tabs(
    ["Overview", "Segment explorer", "Quick scoring"]
)


with tab_overview:
    st.subheader("Overall picture")
    total_customers = len(df_raw)
    churn_rate = (
        df_raw["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of customers", total_customers)
    with col2:
        st.metric("Churn rate (%)", f"{churn_rate:.1f}")

    counts = df_raw["Churn"].value_counts()
    chart_df = counts.rename_axis("Churn").reset_index(name="count")
    st.bar_chart(chart_df.set_index("Churn"))

    st.subheader("Churn by contract type")
    contract_churn = (
        df_raw.groupby("Contract")["Churn"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )
    contract_churn["share"] = contract_churn["share"] * 100
    pivot_cc = contract_churn.pivot(
        index="Contract", columns="Churn", values="share"
    ).fillna(0)
    st.bar_chart(pivot_cc)


with tab_segment:
    st.subheader("Explore a segment")
    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract type", ["All"] + sorted(df_raw["Contract"].unique().tolist())
        )
    with col2:
        tenure_min, tenure_max = int(df_raw["tenure"].min()), int(
            df_raw["tenure"].max()
        )
        tenure_range = st.slider(
            "Tenure (months)", min_value=tenure_min, max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )
    with col3:
        m_min, m_max = float(df_raw["MonthlyCharges"].min()), float(
            df_raw["MonthlyCharges"].max()
        )
        charges_range = st.slider(
            "Monthly charges",
            min_value=float(round(m_min, 1)),
            max_value=float(round(m_max, 1)),
            value=(float(round(m_min, 1)), float(round(m_max, 1))),
        )

    seg = df_raw.copy()
    if contract != "All":
        seg = seg[seg["Contract"] == contract]
    seg = seg[
        (seg["tenure"].between(tenure_range[0], tenure_range[1]))
        & (seg["MonthlyCharges"].between(charges_range[0], charges_range[1]))
    ]

    seg_total = len(seg)
    seg_churn_rate = (
        seg["Churn"].value_counts(normalize=True).get("Yes", 0) * 100
        if seg_total
        else 0
    )
    st.write(f"Customers in segment: **{seg_total}**")
    st.write(f"Churn rate in segment: **{seg_churn_rate:.1f}%**")

    if seg_total:
        seg_counts = seg["Churn"].value_counts()
        seg_chart = (
            seg_counts.rename_axis("Churn").reset_index(name="count")
        )
        st.bar_chart(seg_chart.set_index("Churn"))
        st.dataframe(seg.head(50))
    else:
        st.info("No customers in this segment with current filters.")


with tab_score:
    st.subheader("Quick churn scoring")
    st.write(
        "This is not a full-blown form, just a quick way to see the model's prediction "
        "for a synthetic customer."
    )

    model_path = base / "models" / "modelo_churn.pkl"
    cols_path = base / "models" / "columnas.pkl"
    threshold_path = base / "models" / "threshold.pkl"
    model = None
    feature_cols = None
    threshold = 0.5

    if model_path.exists() and cols_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(cols_path, "rb") as f:
            feature_cols = pickle.load(f)
        if threshold_path.exists():
            with open(threshold_path, "rb") as f:
                threshold = pickle.load(f)
    else:
        st.warning("Model not found. Run python src/train.py first to enable scoring.")

    col1, col2 = st.columns(2)
    with col1:
        contract_q = st.selectbox(
            "Contract",
            sorted(df_raw["Contract"].unique().tolist()),
        )
        tenure_q = st.slider(
            "Tenure (months)",
            min_value=int(df_raw["tenure"].min()),
            max_value=int(df_raw["tenure"].max()),
            value=int(df_raw["tenure"].median()),
        )
    with col2:
        monthly_q = st.slider(
            "Monthly charges",
            min_value=float(round(df_raw["MonthlyCharges"].min(), 1)),
            max_value=float(round(df_raw["MonthlyCharges"].max(), 1)),
            value=float(round(df_raw["MonthlyCharges"].median(), 1)),
        )
        paperless_q = st.selectbox(
            "Paperless billing", sorted(df_raw["PaperlessBilling"].unique().tolist())
        )

    if st.button("Score this synthetic customer"):
        if model is None or feature_cols is None:
            st.error("Model not loaded. Train first, then try again.")
        else:
            sample = {
                "gender": df_raw["gender"].mode()[0],
                "SeniorCitizen": 0,
                "Partner": df_raw["Partner"].mode()[0],
                "Dependents": df_raw["Dependents"].mode()[0],
                "tenure": tenure_q,
                "PhoneService": df_raw["PhoneService"].mode()[0],
                "MultipleLines": df_raw["MultipleLines"].mode()[0],
                "InternetService": df_raw["InternetService"].mode()[0],
                "OnlineSecurity": df_raw["OnlineSecurity"].mode()[0],
                "OnlineBackup": df_raw["OnlineBackup"].mode()[0],
                "DeviceProtection": df_raw["DeviceProtection"].mode()[0],
                "TechSupport": df_raw["TechSupport"].mode()[0],
                "StreamingTV": df_raw["StreamingTV"].mode()[0],
                "StreamingMovies": df_raw["StreamingMovies"].mode()[0],
                "Contract": contract_q,
                "PaperlessBilling": paperless_q,
                "PaymentMethod": df_raw["PaymentMethod"].mode()[0],
                "MonthlyCharges": monthly_q,
                "TotalCharges": monthly_q * max(tenure_q, 1),
                "Churn": "No",
                "customerID": "synthetic-1",
            }
            sample_df = pd.DataFrame([sample])
            sample_df = clean(sample_df)
            sample_df = to_numeric(sample_df)
            X_sample, _ = prepare_for_model(sample_df)

            X_sample = X_sample.reindex(columns=feature_cols, fill_value=0)
            probs = model.predict_proba(X_sample)[0, 1]
            pred = int(probs >= threshold)
            st.write(
                f"Predicted churn: **{pred}** "
                f"(probability: **{probs:.3f}**, threshold: **{threshold:.2f}**)"
            )
