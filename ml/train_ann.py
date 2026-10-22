
# =============================================
# DINEMIND AI JAPAN
# ML-06: ARTIFICIAL NEURAL NETWORK
# DAILY RESTAURANT VISITOR PREDICTION
# =============================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.exceptions import ConvergenceWarning

# STEP 1: PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55, flush=True)
print("DINEMIND AI JAPAN")
print("ML-06: ARTIFICIAL NEURAL NETWORK")
print("=" * 55, flush=True)

# STEP 2: LOAD DATASETS

print("\n[1/6] Loading datasets...", flush=True)

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

assert train["visit_date"].max() < test["visit_date"].min()

print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")

# STEP 3: SELECT FEATURES

print("\n[2/6] Preparing features...", flush=True)

features = [
    "month",
    "day",
    "day_of_week_num",
    "is_weekend",
    "holiday_flg",
    "previous_visitors",
    "previous_7_record_avg",
    "previous_14_record_avg",
    "days_since_previous_record"
]

X_train = train[features]
y_train = train["visitors"]

X_test = test[features]
y_test = test["visitors"]

assert "visitors" not in features
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()
assert np.isfinite(X_train.to_numpy()).all()
assert np.isfinite(X_test.to_numpy()).all()

print("Features selected:", len(features))
print("Training target: visitors")

# STEP 4: BUILD NEURAL NETWORK

print("\n[3/6] Building neural network...", flush=True)

# Architecture:
# 9 inputs -> 32 neurons -> 16 neurons -> 1 output
#
# StandardScaler normalizes numerical inputs.
# TransformedTargetRegressor normalizes the
# visitor-count target during training.

neural_network = Pipeline([
    ("scaler", StandardScaler()),
    ("ann", MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=256,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=True
    ))
])

model = TransformedTargetRegressor(
    regressor=neural_network,
    transformer=StandardScaler()
)

# STEP 5: TRAIN MODEL

print("\n[4/6] Training ANN...", flush=True)
print("This may take several minutes.")
print("Training progress will appear below.\n", flush=True)

with warnings.catch_warnings():
    warnings.simplefilter("default", ConvergenceWarning)
    model.fit(X_train, y_train)

trained_pipeline = model.regressor_
trained_ann = trained_pipeline.named_steps["ann"]

print("\nTraining completed!", flush=True)
print(f"Iterations completed: {trained_ann.n_iter_}")
print(f"Final training loss: {trained_ann.loss_:.6f}")

# STEP 6: EVALUATE MODEL

print("\n[5/6] Evaluating model...", flush=True)

raw_predictions = model.predict(X_test)

# Visitor counts cannot be negative.
predictions = np.maximum(raw_predictions, 0)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(
    y_test,
    predictions
)

# Same simple baseline used for Linear Regression
baseline = X_test["previous_visitors"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline
)

print("\n" + "=" * 55)
print("ARTIFICIAL NEURAL NETWORK RESULTS")
print("=" * 55)

print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print(f"R2 Score:  {r2:.4f}")

print("\nBASELINE COMPARISON")
print(f"Previous-count MAE: {baseline_mae:.4f}")
print(f"ANN MAE:            {mae:.4f}")

# Previous result supplied by Linear Regression
# training. Load its actual recorded metrics.
lr_metrics_file = (
    RESULT_DIR / "linear_regression_metrics.csv"
)

if lr_metrics_file.exists():
    lr_metrics = pd.read_csv(lr_metrics_file)
    lr_mae = float(lr_metrics.iloc[0]["mae"])

    print("\nLINEAR REGRESSION VS ANN")
    print(f"Linear Regression MAE: {lr_mae:.4f}")
    print(f"ANN MAE:               {mae:.4f}")

    if mae < lr_mae:
        print("ANN has lower test MAE.")
    elif mae > lr_mae:
        print("Linear Regression has lower test MAE.")
    else:
        print("Both have equal test MAE.")

    print(
        "\nNote: ANN uses numerical features only, "
        "while the earlier Linear Regression "
        "also uses categorical restaurant features."
    )

# STEP 7: SAVE MODEL AND RESULTS

print("\n[6/6] Saving trained model...", flush=True)

model_file = MODEL_DIR / "ann_model.joblib"

joblib.dump(
    model,
    model_file
)

results = test[
    ["air_store_id", "visit_date", "visitors"]
].copy()

results["predicted_visitors"] = predictions
results["absolute_error"] = np.abs(
    results["visitors"] - results["predicted_visitors"]
)

results.to_csv(
    RESULT_DIR / "ann_results.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "Artificial Neural Network",
    "architecture": "32-16",
    "iterations": trained_ann.n_iter_,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "baseline_mae": baseline_mae,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "ann_metrics.csv",
    index=False
)

# Verify the saved model can be loaded
loaded_model = joblib.load(model_file)

sample = X_test.iloc[[0]]
sample_prediction = loaded_model.predict(sample)[0]

sample_prediction = max(
    0,
    int(round(sample_prediction))
)

actual = int(y_test.iloc[0])

print("\nSAMPLE RESTAURANT PREDICTION")
print("-" * 40)

print("Restaurant:", test.iloc[0]["air_store_id"])
print("Date:", test.iloc[0]["visit_date"].date())
print("Predicted visitors:", sample_prediction)
print("Actual visitors:", actual)

print("\n" + "=" * 55)
print("ML-06 ANN TRAINING COMPLETED!")
print("=" * 55)

print("\nSaved files:")
print("models/ann_model.joblib")
print("results/ann_results.csv")
print("results/ann_metrics.csv")
