from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_roas_model import DATA_PATH, FEATURES, main as train_model, prepare_dataset


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model_outputs" / "best_roas_model.joblib"

app = Flask(__name__)


def load_reference_data() -> tuple[pd.DataFrame, dict, dict]:
    df = prepare_dataset(pd.read_excel(DATA_PATH))

    choices = {
        "Cleaned_Platform": sorted(df["Cleaned_Platform"].dropna().unique().tolist()),
        "Campaign": sorted(df["Campaign"].dropna().unique().tolist()),
        "Region": sorted(df["Region"].dropna().unique().tolist()),
        "Product_Category": sorted(df["Product_Category"].dropna().unique().tolist()),
        "Target_Audience": sorted(df["Target_Audience"].dropna().unique().tolist()),
        "Creative_Type": sorted(df["Creative_Type"].dropna().unique().tolist()),
    }

    defaults = {
        "date": str(df["Date"].max().date()),
        "Cleaned_Platform": df["Cleaned_Platform"].mode().iat[0],
        "Campaign": df["Campaign"].mode().iat[0],
        "Region": df["Region"].mode().iat[0],
        "Spend": round(float(df["Spend"].median()), 2),
        "CPM": round(float(df["CPM"].median()), 2),
        "Impressions": int(df["Impressions"].median()),
        "Frequency": round(float(df["Frequency"].median()), 2),
        "Product_Category": df["Product_Category"].mode().iat[0],
        "Target_Audience": df["Target_Audience"].mode().iat[0],
        "Creative_Type": df["Creative_Type"].mode().iat[0],
        "Customer_LTV": round(float(df["Customer_LTV"].median()), 2),
        "Is_Competitive_Event": False,
        "Cleaned_Completion_Rate": round(float(df["Cleaned_Completion_Rate"].median()), 3),
    }
    return df, choices, defaults


def bootstrap_app() -> tuple[pd.DataFrame, dict, dict, object]:
    if not MODEL_PATH.exists():
        train_model()

    reference_df, form_choices, form_defaults = load_reference_data()
    model = joblib.load(MODEL_PATH)
    return reference_df, form_choices, form_defaults, model


REFERENCE_DF, FORM_CHOICES, FORM_DEFAULTS, MODEL = bootstrap_app()


def build_feature_row(form_data: dict[str, str]) -> pd.DataFrame:
    selected_date = pd.to_datetime(form_data["date"])
    row = {
        "Cleaned_Platform": form_data["Cleaned_Platform"],
        "Campaign": form_data["Campaign"],
        "Region": form_data["Region"],
        "Spend": float(form_data["Spend"]),
        "CPM": float(form_data["CPM"]),
        "Impressions": int(float(form_data["Impressions"])),
        "Frequency": float(form_data["Frequency"]),
        "Product_Category": form_data["Product_Category"],
        "Target_Audience": form_data["Target_Audience"],
        "Creative_Type": form_data["Creative_Type"],
        "Customer_LTV": float(form_data["Customer_LTV"]),
        "Is_Competitive_Event": form_data.get("Is_Competitive_Event") == "true",
        "Cleaned_Completion_Rate": float(form_data["Cleaned_Completion_Rate"]),
        "month": int(selected_date.month),
        "day": int(selected_date.day),
        "dayofweek": int(selected_date.dayofweek),
        "weekofyear": int(selected_date.isocalendar().week),
    }
    return pd.DataFrame([row], columns=FEATURES)


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = FORM_DEFAULTS.copy()
    prediction = None
    error = None

    if request.method == "POST":
        form_values.update(request.form.to_dict())
        try:
            features = build_feature_row(form_values)
            prediction = float(MODEL.predict(features)[0])
        except Exception as exc:  # pragma: no cover
            error = str(exc)

    return render_template(
        "index.html",
        form_values=form_values,
        form_choices=FORM_CHOICES,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, host="127.0.0.1", port=5000)
