
# ==============================================
# DINEMIND AI JAPAN
# ML-05: DECISION TREE CLASSIFIER
# RESTAURANT TRAFFIC CLASSIFICATION
# ==============================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.tree import (
    DecisionTreeClassifier,
    export_text
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: PROJECT FOLDERS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DINEMIND AI JAPAN")
print("ML-05: DECISION TREE")
print("=" * 55, flush=True)

# STEP 2: LOAD DATA

print("\n[1/7] Loading datasets...", flush=True)

train = pd.read_csv(DATA_DIR / "train_data.csv")
test = pd.read_csv(DATA_DIR / "test_data.csv")

train["visit_date"] = pd.to_datetime(
    train["visit_date"]
)
test["visit_date"] = pd.to_datetime(
    test["visit_date"]
)

assert train["visit_date"].max() < test["visit_date"].min()

print(f"Training rows: {len(train):,}")
print(f"Testing rows: {len(test):,}")

# STEP 3: USE EXISTING TRAFFIC THRESHOLDS

print("\n[2/7] Preparing traffic categories...")

thresholds = pd.read_csv(
    MODEL_DIR / "traffic_thresholds.csv"
)

with open(
    MODEL_DIR / "traffic_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

def prepare_labels(df):
    result = df.merge(
        thresholds,
        on="air_store_id",
        how="left",
        validate="many_to_one"
    )

    result["low_limit"] = (
        result["low_limit"]
        .fillna(config["global_low"])
    )

    result["high_limit"] = (
        result["high_limit"]
        .fillna(config["global_high"])
    )

    result["traffic_level"] = np.select(
        [
            result["visitors"] <= result["low_limit"],
            result["visitors"] <= result["high_limit"]
        ],
        ["Low", "Medium"],
        default="High"
    )

    return result

train = prepare_labels(train)
test = prepare_labels(test)

# STEP 4: SELECT FEATURES

print("\n[3/7] Preparing prediction features...")

# All inputs must be available before
# the restaurant-day being predicted.

features = [
    "month",
    "day",
    "day_of_week_num",
    "is_weekend",
    "holiday_flg",
    "previous_visitors",
    "previous_7_record_avg",
    "previous_14_record_avg",
    "days_since_previous_record",
    "low_limit",
    "high_limit"
]

X_train = train[features]
y_train = train["traffic_level"]

X_test = test[features]
y_test = test["traffic_level"]

assert "visitors" not in features
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

print("\nTraining categories:")
print(y_train.value_counts())

# STEP 5: FIND BEST TREE SETTINGS

print("\n[4/7] Tuning Decision Tree...")

validation_start = pd.Timestamp("2017-02-01")

fit_mask = train["visit_date"] < validation_start
val_mask = train["visit_date"] >= validation_start

X_fit = X_train.loc[fit_mask]
y_fit = y_train.loc[fit_mask]

X_val = X_train.loc[val_mask]
y_val = y_train.loc[val_mask]

assert len(X_fit) > 0
assert len(X_val) > 0

candidate_depths = [3, 5, 7, 10, 15]
candidate_leaf_sizes = [20, 50, 100]

best_f1 = -1.0
best_depth = None
best_leaf = None

for depth in candidate_depths:
    for leaf_size in candidate_leaf_sizes:

        candidate = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf_size,
            random_state=42
        )

        candidate.fit(X_fit, y_fit)

        predictions = candidate.predict(X_val)

        score = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0
        )

        print(
            f"Depth={depth:2d}, "
            f"Leaf={leaf_size:3d} | "
            f"Validation Macro F1={score:.4f}",
            flush=True
        )

        if score > best_f1:
            best_f1 = score
            best_depth = depth
            best_leaf = leaf_size

print("\nBest parameters:")
print(f"Max Depth: {best_depth}")
print(f"Min Samples Leaf: {best_leaf}")
print(f"Validation Macro F1: {best_f1:.4f}")

# STEP 6: TRAIN FINAL MODEL

print("\n[5/7] Training final Decision Tree...")

model = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=best_leaf,
    random_state=42
)

model.fit(X_train, y_train)

print("Training completed!")

# STEP 7: EVALUATE

print("\n[6/7] Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test, predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

labels = ["Low", "Medium", "High"]

# Same previous-count baseline as KNN.
baseline = np.select(
    [
        test["previous_visitors"] <= test["low_limit"],
        test["previous_visitors"] <= test["high_limit"]
    ],
    ["Low", "Medium"],
    default="High"
)

baseline_accuracy = accuracy_score(
    y_test, baseline
)

baseline_f1 = f1_score(
    y_test,
    baseline,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 55)
print("DECISION TREE MODEL RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print(f"Macro F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=labels
))

print("Class order: Low, Medium, High")
print("Rows = Actual, Columns = Predicted")

print("\nBaseline Comparison:")
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")
print(f"Decision Tree Accuracy: {accuracy:.2%}")
print(f"Baseline Macro F1: {baseline_f1:.4f}")
print(f"Decision Tree Macro F1: {macro_f1:.4f}")

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nTop Feature Importances:")
print(importance.to_string(index=False))

# STEP 8: SAVE MODEL AND RESULTS

print("\n[7/7] Saving model and results...")

joblib.dump(
    model,
    MODEL_DIR / "decision_tree_model.joblib"
)

results = test[
    ["air_store_id", "visit_date",
     "visitors", "traffic_level"]
].copy()

results["predicted_traffic"] = predictions

results.to_csv(
    RESULT_DIR / "decision_tree_results.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "Decision Tree",
    "best_depth": best_depth,
    "best_min_samples_leaf": best_leaf,
    "validation_macro_f1": best_f1,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "baseline_accuracy": baseline_accuracy,
    "baseline_macro_f1": baseline_f1,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "decision_tree_metrics.csv",
    index=False
)

importance.to_csv(
    RESULT_DIR / "decision_tree_importance.csv",
    index=False
)

# Save learned decision rules.
# Limit display to first 4 levels so
# the explanation is easy to understand.
rules = export_text(
    model,
    feature_names=features,
    max_depth=4
)

with open(
    RESULT_DIR / "decision_tree_rules.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(rules)

# SAMPLE PREDICTION

sample = X_test.iloc[[0]]

predicted = model.predict(sample)[0]

print("\nSAMPLE PREDICTION")
print("-" * 40)

print(
    "Restaurant:",
    test.iloc[0]["air_store_id"]
)
print("Predicted:", predicted)
print("Actual:", y_test.iloc[0])

print("\n" + "=" * 55)
print("ML-05 DECISION TREE COMPLETED!")
print("=" * 55)

print("\nSaved files:")
print("models/decision_tree_model.joblib")
print("results/decision_tree_results.csv")
print("results/decision_tree_metrics.csv")
print("results/decision_tree_importance.csv")
print("results/decision_tree_rules.txt")
