from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_PATH = Path(r"c:\Users\Administrator\Downloads\Training dataset.xlsx")
REPO_DATA_PATH = Path("data") / "Training dataset.xlsx"
DATA_PATH = Path(
    os.getenv(
        "ROAS_DATA_PATH",
        str(REPO_DATA_PATH if REPO_DATA_PATH.exists() else DEFAULT_DATA_PATH),
    )
)
OUTPUT_DIR = Path("model_outputs")
TARGET = "ROAS"

# Only use features that are available before campaign results happen.
FEATURES = [
    "Cleaned_Platform",
    "Campaign",
    "Region",
    "Spend",
    "CPM",
    "Impressions",
    "Frequency",
    "Product_Category",
    "Target_Audience",
    "Creative_Type",
    "Customer_LTV",
    "Is_Competitive_Event",
    "Cleaned_Completion_Rate",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator.astype(float).div(denominator).astype(float)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "Platform": "Cleaned_Platform",
        "Video_Completion_Rate": "Cleaned_Completion_Rate",
    }
    df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

    if "ROAS" not in df.columns and {"Revenue", "Spend"}.issubset(df.columns):
        df["ROAS"] = safe_divide(df["Revenue"], df["Spend"]).fillna(0.0)
    if "CTR" not in df.columns and {"Clicks", "Impressions"}.issubset(df.columns):
        df["CTR"] = safe_divide(df["Clicks"], df["Impressions"]).fillna(0.0)
    if "CPC" not in df.columns and {"Spend", "Clicks"}.issubset(df.columns):
        df["CPC"] = safe_divide(df["Spend"], df["Clicks"]).fillna(0.0)
    if "CVR" not in df.columns and {"Purchase", "Clicks"}.issubset(df.columns):
        df["CVR"] = safe_divide(df["Purchase"], df["Clicks"]).fillna(0.0)
    if "Calculated_CPM" not in df.columns and {"Spend", "Impressions"}.issubset(df.columns):
        df["Calculated_CPM"] = (safe_divide(df["Spend"] * 1000.0, df["Impressions"])).fillna(0.0)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    return df


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return prepare_dataset(df)


def build_quality_summary(df: pd.DataFrame) -> dict:
    metric_validation: dict[str, int | None] = {}

    if {"Clicks", "Impressions", "CTR"}.issubset(df.columns):
        metric_validation["ctr_match_failures"] = int(
            (((safe_divide(df["Clicks"], df["Impressions"]) - df["CTR"]).abs()) > 1e-6).sum()
        )
    if {"Spend", "Clicks", "CPC"}.issubset(df.columns):
        metric_validation["cpc_match_failures"] = int(
            (((safe_divide(df["Spend"], df["Clicks"]) - df["CPC"]).abs()).fillna(0) > 1e-6).sum()
        )
    if {"Purchase", "Clicks", "CVR"}.issubset(df.columns):
        metric_validation["cvr_match_failures"] = int(
            (((safe_divide(df["Purchase"], df["Clicks"]) - df["CVR"]).abs()).fillna(0) > 1e-6).sum()
        )
    if {"Revenue", "Spend", "ROAS"}.issubset(df.columns):
        metric_validation["roas_match_failures"] = int(
            (((safe_divide(df["Revenue"], df["Spend"]) - df["ROAS"]).abs()).fillna(0) > 1e-6).sum()
        )
    if {"Calculated_CPM", "CPM"}.issubset(df.columns):
        metric_validation["cpm_match_failures_gt_0_01"] = int(
            ((df["Calculated_CPM"] - df["CPM"]).abs() > 0.01).sum()
        )

    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
        "missing_values": {
            key: int(value)
            for key, value in df.isna().sum().items()
            if int(value) > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_spend_rows": int((df["Spend"] < 0).sum()),
        "negative_revenue_rows": int((df["Revenue"] < 0).sum()),
        "metric_validation": metric_validation,
        "target_summary": df[TARGET].describe().round(4).to_dict(),
    }


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    x = df[FEATURES]
    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in x.columns if col not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    x_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    x_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    preprocessor = build_preprocessor(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=4,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "RandomForest",
                "rmse": round(float(rmse), 6),
                "mae": round(float(mae), 6),
                "r2": round(float(r2), 6),
            }
        ]
    )
    prediction_df = test_df[["Date", "Campaign", "Cleaned_Platform", "Region", TARGET]].copy()
    prediction_df["predicted_roas"] = predictions
    prediction_df["best_model"] = "RandomForest"
    return metrics_df, prediction_df, pipeline


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def retrain_on_full_data(df: pd.DataFrame, template_pipeline: Pipeline) -> Pipeline:
    final_pipeline = deepcopy(template_pipeline)
    final_pipeline.fit(df[FEATURES], df[TARGET])
    return final_pipeline


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    quality_summary = build_quality_summary(df)
    metrics_df, holdout_predictions_df, best_pipeline = evaluate_models(df)
    feature_importance_df = extract_feature_importance(best_pipeline)
    final_pipeline = retrain_on_full_data(df, best_pipeline)

    joblib.dump(final_pipeline, OUTPUT_DIR / "best_roas_model.joblib")
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    holdout_predictions_df.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    with open(OUTPUT_DIR / "data_quality_summary.json", "w", encoding="utf-8") as file:
        json.dump(quality_summary, file, indent=2)

    print("Saved outputs to:", OUTPUT_DIR.resolve())
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print("\nTop features:")
    print(feature_importance_df.head(10).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
