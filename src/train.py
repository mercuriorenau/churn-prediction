# -*- coding: utf-8 -*-
"""
We train the model and save it so we can use it later without training again.
Also saves visuals: confusion matrix, ROC curve, metrics bar chart, and SHAP summary.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import xgboost as xgb

# Allow imports when run as a script from the project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparar_datos import load, clean, to_numeric, prepare_for_model


def best_threshold(y_true, y_proba):
    """Pick threshold maximizing F1 while keeping churn recall reasonably high."""
    thresholds = np.linspace(0.30, 0.60, 31)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, pred)
        # Soft preference: strong F1, decent recall, not terrible accuracy
        if rec < 0.70:
            continue
        score = f1 + 0.1 * acc
        if score > best_score:
            best_score, best_t = score, float(t)
    if best_score < 0:
        # Fallback if no threshold hits recall floor
        return 0.5, 0.0
    return best_t, best_score


def train(csv_path=None, save_model=True, tune_hyperparams=True):
    df = load(csv_path)
    df = clean(df)
    df = to_numeric(df)
    X, y = prepare_for_model(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Hold out a slice of train to tune the decision threshold (keep test honest)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Balance rare churn class for XGBoost
    pos_weight = (y_fit == 0).sum() / max((y_fit == 1).sum(), 1)
    base_clf = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
    )

    if tune_hyperparams:
        param_grid = {
            "n_estimators": [150, 250, 400],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.08, 0.12],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
            "gamma": [0, 0.5, 1.0],
        }
        # Optimize for churn F1, not overall accuracy
        search = RandomizedSearchCV(
            base_clf,
            param_grid,
            n_iter=40,
            scoring="f1",
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_fit, y_fit)
        model = search.best_estimator_
        print("Best params (F1):", search.best_params_)
        print("Best CV F1:", round(search.best_score_, 4))
    else:
        model = base_clf
        model.fit(X_fit, y_fit)

    # Threshold from validation only; then refit on full training data
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = best_threshold(y_val, val_proba)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"Decision threshold (tuned for F1): {threshold:.2f}")
    print(classification_report(y_test, y_pred))
    print("F1 (churn):", round(f1_score(y_test, y_pred, pos_label=1), 4))
    print("Recall (churn):", round(recall_score(y_test, y_pred, pos_label=1), 4))
    print("AUC:", round(roc_auc_score(y_test, y_proba), 4))

    base = Path(__file__).resolve().parent.parent
    reports_dir = base / "reports"
    reports_dir.mkdir(exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (XGBoost)")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved reports/confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()
    print("Saved reports/roc_curve.png")

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    prec = precision_score(y_test, y_pred, pos_label=1)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1", "Recall (churn)", "Precision (churn)"],
        "Value": [acc, f1, rec, prec],
    })
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=metrics_df, x="Metric", y="Value", color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model metrics (test set)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(reports_dir / "metrics_summary.png", dpi=150)
    plt.close()
    print("Saved reports/metrics_summary.png")

    # SHAP summary on a capped test sample (keeps train runtime reasonable)
    shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(shap_sample)
    plt.figure()
    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.title("SHAP feature impact on churn prediction")
    plt.tight_layout()
    plt.savefig(reports_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved reports/shap_summary.png")

    # Persist comparison row for the README / reports
    xgb_row = {
        "Model": "XGBoost (tuned)",
        "Accuracy": round(acc, 2),
        "F1": round(f1, 2),
        "Recall (churn)": round(rec, 2),
        "Precision (churn)": round(prec, 2),
        "AUC": round(auc, 2),
        "Threshold": round(threshold, 2),
    }
    comparison_path = reports_dir / "model_comparison.csv"
    if comparison_path.exists():
        prev = pd.read_csv(comparison_path)
        prev = prev[~prev["Model"].astype(str).str.contains("XGBoost", case=False)]
        for col in ("Precision (churn)", "AUC", "Threshold"):
            if col not in prev.columns:
                prev[col] = ""
        comparison = pd.concat([prev, pd.DataFrame([xgb_row])], ignore_index=True)
    else:
        comparison = pd.DataFrame([xgb_row])
    comparison.to_csv(comparison_path, index=False)
    print("Updated reports/model_comparison.csv")

    if save_model:
        (base / "models").mkdir(exist_ok=True)
        with open(base / "models" / "modelo_churn.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(base / "models" / "columnas.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)
        with open(base / "models" / "threshold.pkl", "wb") as f:
            pickle.dump(threshold, f)
        print("Model saved to models/modelo_churn.pkl")
        print(f"Threshold saved ({threshold:.2f})")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    train()
