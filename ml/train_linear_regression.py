
# ==============================================
# DINEMIND AI JAPAN
# ML-01: LINEAR REGRESSION
# RESTAURANT DAILY VISITOR PREDICTION
# ==============================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# ----------------------------------------------
# STEP 1: PROJECT PATHS
# ----------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DINEMIND AI JAPAN")
print("ML-01: LINEAR REGRESSION")
print("=" * 55)

# ----------------------------------------------
# STEP 2: LOAD TRAINING AND TEST DATA
# ----------------------------------------------

print("\n[1/6] Loading prepared datasets...")

train = pd.read_csv(
    DATA_DIR / "train_data.csv"
)

test = pd.read_csv(
    DATA_DIR / "test_data.csv"
)

train["visit_date"] = pd.to_datetime(
    train["visit_date"]
)

test["visit_date"] = pd.to_datetime(
    test["visit_date"]
)

assert train["visit_date"].max() < test["visit_date"].min(), (
    "Training and testing dates overlap!"
)

print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")

# ----------------------------------------------
# STEP 3: SELECT FEATURES
# ----------------------------------------------

print("\n[2/6] Selecting model features...")

numeric_features = [
    "month",
    "day",
    "day_of_week_num",
    "is_weekend",
    "holiday_flg",
    "previous_visitors",
    "previous_7_record_avg",
    "previous_14_record_avg",
    "days_since_previous_record",
]

categorical_features = [
    "air_store_id",
    "air_genre_name",
    "air_area_name",
]

feature_columns = (
    numeric_features + categorical_features
)

target_column = "visitors"

# Prevent actual visitor counts from entering
# the prediction input directly.
assert target_column not in feature_columns

X_train = train[feature_columns]
y_train = train[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

assert not X_train.isna().any().any(), (
    "Missing training feature values!"
)

assert not X_test.isna().any().any(), (
    "Missing testing feature values!"
)

# ----------------------------------------------
# STEP 4: BUILD AND TRAIN MODEL
# ----------------------------------------------

print("\n[3/6] Training Linear Regression...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

model.fit(X_train, y_train)

print("Model training completed!")

# ----------------------------------------------
# STEP 5: EVALUATE MODEL
# ----------------------------------------------

print("\n[4/6] Evaluating model performance...")

raw_predictions = model.predict(X_test)

# Clip negative predictions because
# negative customer counts are impossible.
predictions = np.maximum(raw_predictions, 0)

mae = mean_absolute_error(
    y_test, predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

# Baseline: simply predict that today's
# visitor count equals the previous recorded count.
baseline_predictions = X_test[
    "previous_visitors"
].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

print("\n" + "=" * 55)
print("LINEAR REGRESSION MODEL RESULTS")
print("=" * 55)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nBaseline comparison:")
print(f"Previous-count baseline MAE: {baseline_mae:.4f}")
print(f"Linear Regression MAE: {mae:.4f}")

if mae < baseline_mae:
    print("Result: Model beats the simple baseline.")
else:
    print("Result: Baseline performs as well or better.")

# ----------------------------------------------
# STEP 6: SAVE TRAINED MODEL
# ----------------------------------------------

print("\n[5/6] Saving trained model...")

model_path = (
    MODEL_DIR / "linear_regression_model.joblib"
)

joblib.dump(model, model_path)

print(f"Model saved: {model_path}")

# Save full prediction results
results = test[
    ["air_store_id", "visit_date", "visitors"]
].copy()

results["predicted_visitors"] = predictions
results["absolute_error"] = np.abs(
    results["visitors"] -
    results["predicted_visitors"]
)

results.to_csv(
    RESULT_DIR / "linear_regression_results.csv",
    index=False
)

# Save numerical evaluation metrics
metrics = pd.DataFrame([{
    "model": "Linear Regression",
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "baseline_mae": baseline_mae,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "linear_regression_metrics.csv",
    index=False
)

# ----------------------------------------------
# STEP 7: TEST A SAMPLE PREDICTION
# ----------------------------------------------

print("\n[6/6] Testing a sample prediction...")

# Use a real historical test observation
sample = X_test.iloc[[0]]

sample_prediction = model.predict(sample)[0]

sample_prediction = max(
    0, int(round(sample_prediction))
)

actual_visitors = int(y_test.iloc[0])

print("\nSAMPLE RESTAURANT PREDICTION")
print("-" * 40)

print(
    f"Restaurant: {sample.iloc[0]['air_store_id']}"
)

print(
    f"Previous visitors: "
    f"{sample.iloc[0]['previous_visitors']:.0f}"
)

print(
    f"Predicted visitors: {sample_prediction}"
)

print(
    f"Actual visitors: {actual_visitors}"
)

print("\n" + "=" * 55)
print("ML-01 TRAINING COMPLETED!")
print("=" * 55)

print("\nGenerated files:")
print("1. models/linear_regression_model.joblib")
print("2. results/linear_regression_results.csv")
print("3. results/linear_regression_metrics.csv")

print("\nNEXT: REVIEW PERFORMANCE AND BUILD PREDICTION UI")
