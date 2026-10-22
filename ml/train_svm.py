
# ==============================================
# DINEMIND AI JAPAN
# ML-04: SUPPORT VECTOR MACHINE
# HIGH-TRAFFIC RESTAURANT CLASSIFICATION
# ==============================================

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: SET UP PATHS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DINEMIND AI JAPAN")
print("ML-04: SUPPORT VECTOR MACHINE")
print("=" * 55)

# STEP 2: LOAD DATASETS

print("\n[1/7] Loading restaurant datasets...")

train = pd.read_csv(DATA_DIR / "train_data.csv")
test = pd.read_csv(DATA_DIR / "test_data.csv")

train["visit_date"] = pd.to_datetime(train["visit_date"])
test["visit_date"] = pd.to_datetime(test["visit_date"])

assert train["visit_date"].max() < test["visit_date"].min(), (
    "Training dates overlap with testing dates!"
)

print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")

# STEP 3: LOAD EXISTING TRAFFIC THRESHOLDS

print("\n[2/7] Loading traffic thresholds...")

thresholds = pd.read_csv(
    MODEL_DIR / "traffic_thresholds.csv"
)

with open(
    MODEL_DIR / "traffic_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

def add_labels(df):
    result = df.merge(
        thresholds,
        on="air_store_id",
        how="left",
        validate="many_to_one"
    )

    result["high_limit"] = (
        result["high_limit"]
        .fillna(config["global_high"])
    )

    # Match the earlier Low / Medium / High
    # classification boundaries:
    # High means visitors > high_limit.
    result["high_traffic"] = (
        result["visitors"] > result["high_limit"]
    ).astype(int)

    return result

train = add_labels(train)
test = add_labels(test)

print("\nTraining target distribution:")
print(train["high_traffic"].value_counts())

print("\n0 = Not High Traffic")
print("1 = High Traffic")

# STEP 4: SELECT FEATURES

print("\n[3/7] Preparing model features...")

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
y_train = train["high_traffic"]

X_test = test[features]
y_test = test["high_traffic"]

assert "visitors" not in features
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

# STEP 5: VALIDATION AND HYPERPARAMETER SELECTION

print("\n[4/7] Selecting SVM settings...")

validation_start = pd.Timestamp("2017-02-01")

fit_mask = train["visit_date"] < validation_start
val_mask = train["visit_date"] >= validation_start

X_fit = X_train.loc[fit_mask]
y_fit = y_train.loc[fit_mask]

X_val = X_train.loc[val_mask]
y_val = y_train.loc[val_mask]

assert len(X_fit) > 0 and len(X_val) > 0

candidate_c = [0.01, 0.1, 1.0]

best_c = None
best_f1 = -1.0

for c_value in candidate_c:

    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=False,
            max_iter=5000,
            random_state=42
        ))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        candidate.fit(X_fit, y_fit)

    val_predictions = candidate.predict(X_val)

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    print(
        f"C={c_value} | "
        f"Validation F1={score:.4f}"
    )

    if score > best_f1:
        best_f1 = score
        best_c = c_value

print(f"\nSelected C: {best_c}")
print(f"Best validation F1: {best_f1:.4f}")

# STEP 6: TRAIN FINAL MODEL

print("\n[5/7] Training final SVM model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=best_c,
        class_weight="balanced",
        dual=False,
        max_iter=5000,
        random_state=42
    ))
])

model.fit(X_train, y_train)

print("SVM training completed!")

# STEP 7: EVALUATE THE MODEL

print("\n[6/7] Evaluating SVM performance...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

# Simple baseline:
# Classify as high if previous recorded
# visitor count exceeds the restaurant's
# high-traffic threshold.
baseline_predictions = (
    test["previous_visitors"] >
    test["high_limit"]
).astype(int)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0
)

print("\n" + "=" * 55)
print("SVM MODEL RESULTS")
print("=" * 55)

print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=[
        "Not High Traffic",
        "High Traffic"
    ],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
))

print("Rows = Actual, Columns = Predicted")
print("Class order: [Not High, High]")

print("\nBaseline Comparison:")
print(f"Baseline accuracy: {baseline_accuracy:.2%}")
print(f"SVM accuracy:      {accuracy:.2%}")

print(f"Baseline F1: {baseline_f1:.4f}")
print(f"SVM F1:      {f1:.4f}")

# Also compare with always predicting
# Not High Traffic.
majority_predictions = np.zeros(len(y_test), dtype=int)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

print(
    f"Always Not High accuracy: "
    f"{majority_accuracy:.2%}"
)

# STEP 8: SAVE MODEL AND RESULTS

print("\n[7/7] Saving model and results...")

joblib.dump(
    model,
    MODEL_DIR / "svm_model.joblib"
)

results = test[
    ["air_store_id", "visit_date", "visitors"]
].copy()

results["actual_high_traffic"] = y_test.to_numpy()
results["predicted_high_traffic"] = predictions

results.to_csv(
    RESULT_DIR / "svm_results.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "Linear Support Vector Machine",
    "best_c": best_c,
    "validation_f1": best_f1,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "baseline_accuracy": baseline_accuracy,
    "baseline_f1": baseline_f1,
    "majority_accuracy": majority_accuracy,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "svm_metrics.csv",
    index=False
)

# Sample historical prediction
sample = X_test.iloc[[0]]
sample_prediction = int(model.predict(sample)[0])

print("\nSAMPLE PREDICTION")
print("-" * 40)
print("Restaurant:", test.iloc[0]["air_store_id"])
print(
    "Predicted:",
    "High Traffic" if sample_prediction == 1
    else "Not High Traffic"
)

print(
    "Actual:",
    "High Traffic" if int(y_test.iloc[0]) == 1
    else "Not High Traffic"
)

print("\n" + "=" * 55)
print("ML-04 SVM TRAINING COMPLETED!")
print("=" * 55)

print("\nGenerated files:")
print("models/svm_model.joblib")
print("results/svm_results.csv")
print("results/svm_metrics.csv")
