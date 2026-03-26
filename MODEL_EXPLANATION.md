# ROAS Predictive Model

This project treats the task as a **supervised regression problem** because the target variable, `ROAS`, is a continuous number.

## Why regression is suitable

`ROAS = Revenue / Spend`, so we want the model to estimate the expected return for a campaign configuration. That makes regression the correct problem type.

The model used here is **Random Forest Regression**.

Random Forest is a strong choice because:

- it handles nonlinear relationships well
- it works well on structured marketing data
- it builds many decision trees and combines them for a more stable prediction
- it is less fragile than a single decision tree

On the time-based holdout split, the Random Forest model achieved:

- RMSE `0.2260`
- MAE `0.1757`
- R² `0.5202`

## Important modeling decision

Some columns should **not** be used as predictors for forecasting:

- `Revenue`
- `Purchase`
- `CTR`
- `CPC`
- `CVR`
- `AOV`
- `ROAS`

These are outcome or post-campaign metrics. If they are included, the model gets information from the future and becomes unrealistically strong.

## Features used by the model

The model uses inputs that are known or reasonably planned before evaluating a campaign:

- Platform
- Campaign
- Region
- Spend
- CPM
- Impressions
- Frequency
- Product category
- Target audience
- Creative type
- Customer LTV
- Competitive event flag
- Completion rate
- Calendar features derived from date

## How the model works

1. The Excel dataset is loaded and sorted by date.
2. Date features are created: month, day, day of week, and week of year.
3. Numeric fields are median-imputed and scaled.
4. Categorical fields are imputed and one-hot encoded.
5. A **Random Forest regressor** is trained using many decision trees.
6. The data is split chronologically:
   - First 80% for training
   - Last 20% for testing
7. The trained model is evaluated on the holdout period.
8. The final Random Forest model is retrained on the full dataset and saved.

## How this helps the business

The model estimates expected ROAS for different campaign setups. The marketing team can use it to:

- rank campaigns before scaling budget
- identify low-expected-ROAS combinations to cut
- compare likely performance by platform, region, creative, and audience
- plan spend more carefully during competitive events

## Files produced after running the script

- `model_outputs/best_roas_model.joblib`: trained model
- `model_outputs/model_comparison.csv`: comparison of candidate regressors
- `model_outputs/holdout_predictions.csv`: actual vs predicted ROAS on the holdout set
- `model_outputs/feature_importance.csv`: most influential features
- `model_outputs/data_quality_summary.json`: data quality checks

## How to run

```powershell
python train_roas_model.py
```
