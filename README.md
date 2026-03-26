# Marketing ROAS Predictor

A Flask-based machine learning project that predicts Return on Ad Spend (ROAS) for digital marketing campaigns.

This project includes:

- a training pipeline for building a ROAS prediction model from Excel campaign data
- a Flask web app for interactive predictions
- model evaluation outputs and feature importance analysis
- data normalization so the app can work with raw columns like `Platform` and `Video_Completion_Rate`

## Project Structure

```text
Marketing/
|-- app.py
|-- train_roas_model.py
|-- MODEL_EXPLANATION.md
|-- README.md
|-- requirements.txt
|-- templates/
|   `-- index.html
`-- model_outputs/
```

## Features

- Predicts campaign ROAS using a Random Forest regressor
- Uses time-based train/test splitting for more realistic evaluation
- Builds a browser-based prediction form with Flask
- Auto-computes missing derived marketing metrics such as `ROAS`, `CTR`, `CPC`, and `CVR`
- Auto-trains the model on first Flask app launch if the model file is missing

## Tech Stack

- Python
- Flask
- Pandas
- scikit-learn
- Joblib
- OpenPyXL

## Dataset Requirements

The training data should be an Excel file with columns similar to:

- `Date`
- `Platform`
- `Campaign`
- `Region`
- `Spend`
- `CPM`
- `Impressions`
- `Frequency`
- `Clicks`
- `Purchase`
- `Revenue`
- `Product_Category`
- `Target_Audience`
- `Creative_Type`
- `Video_Completion_Rate`
- `Customer_LTV`
- `Is_Competitive_Event`

The code automatically maps:

- `Platform` -> `Cleaned_Platform`
- `Video_Completion_Rate` -> `Cleaned_Completion_Rate`

## Setup

1. Clone the repository
2. Install dependencies
3. Point the app to your Excel dataset
4. Run the Flask app

```powershell
git clone <your-repo-url>
cd Marketing
pip install -r requirements.txt
```

Set your dataset path before running:

```powershell
$env:ROAS_DATA_PATH="C:\path\to\Training dataset.xlsx"
```

## Run The App

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Train The Model Manually

```powershell
python train_roas_model.py
```

This creates:

- `model_outputs/best_roas_model.joblib`
- `model_outputs/model_comparison.csv`
- `model_outputs/holdout_predictions.csv`
- `model_outputs/feature_importance.csv`
- `model_outputs/data_quality_summary.json`

## Model Summary

The current approach treats ROAS prediction as a supervised regression problem using a Random Forest model. The app uses only planning-time campaign features and avoids post-outcome leakage from columns such as final revenue-driven metrics.

More detail is available in `MODEL_EXPLANATION.md`.

## Notes For GitHub

- Do not commit private datasets
- Generated model artifacts are ignored by default
- The dataset path is configurable with the `ROAS_DATA_PATH` environment variable

## License

This project is available for portfolio and educational use.
