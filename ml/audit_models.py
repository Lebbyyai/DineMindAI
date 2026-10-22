
# ==========================================
# DINEMIND AI JAPAN
# PHASE 3: MACHINE LEARNING MODEL AUDIT
# ==========================================

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "datasets" / "processed"
MODELS = BASE / "models"
RESULTS = BASE / "results"

RESULTS.mkdir(exist_ok=True)

print("=" * 55)
print("DINEMIND AI - MODEL AUDIT")
print("=" * 55)

train = pd.read_csv(DATA / "train_data.csv")
test = pd.read_csv(DATA / "test_data.csv")

train["visit_date"] = pd.to_datetime(train["visit_date"])
test["visit_date"] = pd.to_datetime(test["visit_date"])

# ------------------------------------------
# 1. DATASET VALIDATION
# ------------------------------------------

print("\n[1/4] Checking dataset integrity...")

checks = {
    "Training data not empty": len(train) > 0,
    "Testing data not empty": len(test) > 0,
    "Chronological split": (
        train["visit_date"].max()
        < test["visit_date"].min()
    ),
    "No duplicate training records": (
        not train.duplicated(
            ["air_store_id", "visit_date"]
        ).any()
    ),
    "No duplicate testing records": (
        not test.duplicated(
            ["air_store_id", "visit_date"]
        ).any()
    ),
    "No missing visitor counts": (
        train["visitors"].notna().all()
        and test["visitors"].notna().all()
    ),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

# ------------------------------------------
# 2. CHECK PREPARED HISTORICAL FEATURES
# ------------------------------------------

print("\n[2/4] Checking historical features...")

master = pd.read_csv(
    DATA / "master_restaurant_data.csv",
    parse_dates=["visit_date"]
)

master = master.sort_values(
    ["air_store_id", "visit_date"]
).reset_index(drop=True)

expected_previous = (
    master.groupby("air_store_id")["visitors"]
    .shift(1)
)

previous_check = np.allclose(
    master["previous_visitors"].to_numpy(),
    expected_previous.to_numpy(),
    equal_nan=True
)

print(
    "PASS" if previous_check else "FAIL",
    "- Previous visitors use shifted records"
)

# ------------------------------------------
# 3. LOAD SAVED MODELS
# ------------------------------------------

print("\n[3/4] Loading and testing saved models...")

model_files = {
    "Linear Regression": "linear_regression_model.joblib",
    "Naive Bayes": "naive_bayes_model.joblib",
    "KNN": "knn_model.joblib",
    "SVM": "svm_model.joblib",
    "Decision Tree": "decision_tree_model.joblib",
    "ANN": "ann_model.joblib"
}

numeric_features = [
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

feature_sets = {
    "Linear Regression": numeric_features + [
        "air_store_id",
        "air_genre_name",
        "air_area_name"
    ],
    "Naive Bayes": numeric_features + [
        "air_store_id",
        "air_genre_name"
    ],
    "KNN": [
        "month",
        "day_of_week_num",
        "is_weekend",
        "holiday_flg",
        "previous_visitors",
        "previous_7_record_avg",
        "previous_14_record_avg",
        "days_since_previous_record"
    ],
    "SVM": numeric_features,
    "Decision Tree": numeric_features + [
        "low_limit",
        "high_limit"
    ],
    "ANN": numeric_features
}

# Load threshold definitions for Decision Tree
thresholds = pd.read_csv(
    MODELS / "traffic_thresholds.csv"
)

with open(
    MODELS / "traffic_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

test_with_thresholds = test.merge(
    thresholds,
    on="air_store_id",
    how="left",
    validate="many_to_one"
)

for col, fallback in [
    ("low_limit", config["global_low"]),
    ("high_limit", config["global_high"])
]:
    test_with_thresholds[col] = (
        test_with_thresholds[col].fillna(fallback)
    )

audit_results = []

for name, filename in model_files.items():
    path = MODELS / filename

    print(f"\n{name}:")

    if not path.exists():
        print("  FAIL - Model file missing")
        audit_results.append({
            "model": name,
            "status": "MISSING",
            "details": filename
        })
        continue

    try:
        # Load only model files created by our
        # own trusted training scripts.
        model = joblib.load(path)

        features = feature_sets[name]

        sample = test_with_thresholds[
            features
        ].head(5).copy()

        assert not sample.isna().any().any()

        predictions = model.predict(sample)

        assert len(predictions) == 5

        print("  PASS - Model loaded")
        print("  PASS - Predictions generated")
        print("  Features:", len(features))
        print("  Sample output:", predictions[0])

        audit_results.append({
            "model": name,
            "status": "PASS",
            "details": "Loaded and predicted 5 samples"
        })

    except Exception as error:
        print("  FAIL:", error)
        audit_results.append({
            "model": name,
            "status": "FAIL",
            "details": str(error)
        })

# ------------------------------------------
# 4. SAVE REPORT
# ------------------------------------------

print("\n[4/4] Saving audit report...")

report = pd.DataFrame(audit_results)

report.to_csv(
    RESULTS / "model_audit_report.csv",
    index=False
)

print("\n" + "=" * 55)
print("MODEL AUDIT SUMMARY")
print("=" * 55)

print(report[["model", "status"]].to_string(index=False))

print("\nReport: results/model_audit_report.csv")

print("\nImportant:")
print("Passing this audit confirms basic compatibility,")
print("not that the models are free from all data leakage.")
print("Prediction-time and evaluation checks remain necessary.")
