
# ============================================
# DINEMIND AI JAPAN
# ML-03: K-NEAREST NEIGHBORS
# RESTAURANT TRAFFIC CLASSIFICATION
# ============================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DINEMIND AI JAPAN")
print("ML-03: K-NEAREST NEIGHBORS")
print("=" * 55)

# 1. LOAD DATA
print("\n[1/7] Loading datasets...")

train = pd.read_csv(DATA_DIR / "train_data.csv")
test = pd.read_csv(DATA_DIR / "test_data.csv")

train["visit_date"] = pd.to_datetime(train["visit_date"])
test["visit_date"] = pd.to_datetime(test["visit_date"])

assert train["visit_date"].max() < test["visit_date"].min()

print(f"Training rows: {len(train):,}")
print(f"Testing rows: {len(test):,}")

# 2. LOAD THE SAME THRESHOLDS AS NAIVE BAYES
print("\n[2/7] Loading traffic categories...")

thresholds = pd.read_csv(
    MODEL_DIR / "traffic_thresholds.csv"
)

with open(
    MODEL_DIR / "traffic_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

def label_traffic(df):
    result = df.merge(
        thresholds,
        on="air_store_id",
        how="left",
        validate="many_to_one"
    )

    low = result["low_limit"].fillna(
        config["global_low"]
    )
    high = result["high_limit"].fillna(
        config["global_high"]
    )

    result["traffic_level"] = np.select(
        [
            result["visitors"] <= low,
            result["visitors"] <= high
        ],
        ["Low", "Medium"],
        default="High"
    )

    result["low_limit"] = low
    result["high_limit"] = high

    return result

train = label_traffic(train)
test = label_traffic(test)

# 3. SELECT NUMERICAL FEATURES
print("\n[3/7] Preparing features...")

features = [
    "month",
    "day_of_week_num",
    "is_weekend",
    "holiday_flg",
    "previous_visitors",
    "previous_7_record_avg",
    "previous_14_record_avg",
    "days_since_previous_record"
]

X_train_all = train[features]
y_train_all = train["traffic_level"]

X_test = test[features]
y_test = test["traffic_level"]

assert not X_train_all.isna().any().any()
assert not X_test.isna().any().any()

# 4. TEMPORAL VALIDATION
# Tune K before evaluating the final test set.
print("\n[4/7] Selecting the best K...")

validation_start = pd.Timestamp("2017-02-01")

fit_mask = train["visit_date"] < validation_start
val_mask = train["visit_date"] >= validation_start

X_fit = X_train_all.loc[fit_mask]
y_fit = y_train_all.loc[fit_mask]

X_val = X_train_all.loc[val_mask]
y_val = y_train_all.loc[val_mask]

assert len(X_fit) > 0 and len(X_val) > 0

candidate_k = [5, 11, 21, 31, 51]
best_k = None
best_f1 = -1

for k in candidate_k:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",
            n_jobs=-1
        ))
    ])

    candidate.fit(X_fit, y_fit)
    val_predictions = candidate.predict(X_val)

    score = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    print(f"K={k:2d} | Validation Macro F1={score:.4f}")

    if score > best_f1:
        best_f1 = score
        best_k = k

print(f"\nSelected K: {best_k}")

# 5. TRAIN FINAL MODEL
print("\n[5/7] Training final KNN model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=best_k,
        weights="distance",
        n_jobs=-1
    ))
])

model.fit(X_train_all, y_train_all)

# 6. EVALUATE FINAL TEST
print("\n[6/7] Evaluating test performance...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

labels = ["Low", "Medium", "High"]

baseline = np.select(
    [
        test["previous_visitors"] <= test["low_limit"],
        test["previous_visitors"] <= test["high_limit"]
    ],
    ["Low", "Medium"],
    default="High"
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline
)

print("\n" + "=" * 55)
print("KNN MODEL RESULTS")
print("=" * 55)

print(f"Best K: {best_k}")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=labels
))

print("\nBaseline Comparison:")
print(f"Previous-count baseline: {baseline_accuracy:.2%}")
print(f"Naive Bayes accuracy: 35.62%")
print(f"KNN accuracy: {accuracy:.2%}")

# 7. SAVE MODEL AND RESULTS
print("\n[7/7] Saving results...")

joblib.dump(
    model,
    MODEL_DIR / "knn_model.joblib"
)

results = test[
    ["air_store_id", "visit_date", "visitors", "traffic_level"]
].copy()

results["predicted_traffic"] = predictions

results.to_csv(
    RESULT_DIR / "knn_results.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "K-Nearest Neighbors",
    "best_k": best_k,
    "validation_macro_f1": best_f1,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "baseline_accuracy": baseline_accuracy,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "knn_metrics.csv",
    index=False
)

print("\nML-03 KNN TRAINING COMPLETED!")
print("Model: models/knn_model.joblib")
print("Results: results/knn_results.csv")
print("Metrics: results/knn_metrics.csv")
