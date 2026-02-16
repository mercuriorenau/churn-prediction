# -*- coding: utf-8 -*-
"""
Load the saved model and predict churn for new data. Run after train.py or the model notebook.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preparar_datos import load, clean, to_numeric, prepare_for_model


def load_model():
    model_path = ROOT / "models" / "modelo_churn.pkl"
    cols_path = ROOT / "models" / "columnas.pkl"
    threshold_path = ROOT / "models" / "threshold.pkl"
    if not model_path.exists():
        print("No saved model found. Run train.py or run the model notebook first, then try again.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cols_path, "rb") as f:
        expected_cols = pickle.load(f)
    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
    return model, expected_cols, threshold


def predict(csv_path=None, model=None, expected_cols=None, threshold=0.5):
    if model is None or expected_cols is None:
        model, expected_cols, threshold = load_model()
    if csv_path is None:
        data_dir = ROOT / "data"
        csvs = list(data_dir.glob("*.csv"))
        if not csvs:
            print("No CSV in data/. Put the Telco CSV there or pass a path: python src/predict.py --input path/to/file.csv")
            sys.exit(1)
        csv_path = csvs[0]
    csv_path = Path(csv_path)

    df = load(csv_path)
    df = clean(df)
    df = to_numeric(df)
    X, _ = prepare_for_model(df)

    # Match training feature set; fill missing columns with 0
    missing = set(expected_cols) - set(X.columns)
    if missing:
        print("Warning: some columns the model needs are missing. Filling with 0:", list(missing)[:5])
    X = X.reindex(columns=expected_cols, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    df_out = df[["customerID"]].copy() if "customerID" in df.columns else df.index.to_series().reset_index(drop=True)
    df_out = df_out.iloc[: len(pred)]
    df_out["churn_pred"] = pred
    df_out["churn_probability"] = proba.round(3)

    out_path = ROOT / "reports" / "predictions.csv"
    out_path.parent.mkdir(exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print("Predictions saved to", out_path)
    print(f"Threshold used: {threshold:.2f}")
    print("Churn rate (predicted):", pred.mean().round(2), "| Sample of first rows:")
    print(df_out.head(10).to_string())
    return df_out


if __name__ == "__main__":
    model, expected_cols, threshold = load_model()
    args = sys.argv[1:]
    csv_path = None
    if args and args[0] == "--input" and len(args) > 1:
        csv_path = args[1]
    predict(csv_path, model=model, expected_cols=expected_cols, threshold=threshold)
