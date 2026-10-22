
# ==========================================
# DINEMIND AI JAPAN
# ML-02: GAUSSIAN NAIVE BAYES
# RESTAURANT TRAFFIC CLASSIFICATION
# ==========================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# STEP 1: FOLDERS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("DINEMIND AI JAPAN")
print("ML-02: NAIVE BAYES TRAFFIC CLASSIFICATION")
print("=" * 55)

# STEP 2: LOAD DATA

print("\n[1/6] Loading datasets...")

train = pd.read_csv(DATA_DIR / "train_data.csv")
test = pd.read_csv(DATA_DIR / "test_data.csv")

train["visit_date"] = pd.to_datetime(train["visit_date"])
test["visit_date"] = pd.to_datetime(test["visit_date"])

assert train["visit_date"].max() < test["visit_date"].min()

print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")

# STEP 3: DEFINE TRAFFIC CATEGORIES
# Thresholds come from training data ONLY.
# Each restaurant has its own traffic levels.

print("\n[2/6] Defining traffic categories...")

thresholds = (
    train.groupby("air_store_id")["visitors"]
    .quantile([1/3, 2/3])
    .unstack()
)

thresholds.columns = ["low_limit", "high_limit"]

# Handle restaurants present only in test data
global_low = train["visitors"].quantile(1/3)
global_high = train["visitors"].quantile(2/3)

def add_traffic_labels(df):
    result = df.join(
        thresholds,
        on="air_store_id"
    )

    result["low_limit"] = (
        result["low_limit"].fillna(global_low)
    )

    result["high_limit"] = (
        result["high_limit"].fillna(global_high)
    )

    conditions = [
        result["visitors"] <= result["low_limit"],
        result["visitors"] <= result["high_limit"]
    ]

    result["traffic_level"] = np.select(
        conditions,
        ["Low", "Medium"],
        default="High"
    )

    return result

train = add_traffic_labels(train)
test = add_traffic_labels(test)

print("\nTraining class distribution:")
print(train["traffic_level"].value_counts())

# STEP 4: FEATURES AND PREPROCESSING

print("\n[3/6] Preparing model features...")

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

categorical_features = [
    "air_store_id",
    "air_genre_name"
]

features = numeric_features + categorical_features

X_train = train[features]
y_train = train["traffic_level"]

X_test = test[features]
y_test = test["traffic_level"]

assert not X_train.isna().any().any()
assert not X_test.isna().any().any()

# Sparse output disabled for GaussianNB.
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
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    sparse_threshold=0
)

# STEP 5: TRAIN NAIVE BAYES

print("\n[4/6] Training Gaussian Naive Bayes...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB(var_smoothing=1e-7))
])

model.fit(X_train, y_train)

print("Training completed!")

# STEP 6: EVALUATE

print("\n[5/6] Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

labels = ["Low", "Medium", "High"]

print("\n" + "=" * 55)
print("NAIVE BAYES MODEL RESULTS")
print("=" * 55)

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
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
print("Rows = Actual, Columns = Predicted")
print("Class order:", labels)

# Baseline: predict traffic using the
# previous recorded visitor count and the
# same restaurant-specific thresholds.

baseline = np.select(
    [
        test["previous_visitors"] <= test["low_limit"],
        test["previous_visitors"] <= test["high_limit"]
    ],
    ["Low", "Medium"],
    default="High"
)

baseline_accuracy = accuracy_score(y_test, baseline)

print("\nBaseline comparison:")
print(f"Previous-count baseline: {baseline_accuracy:.2%}")
print(f"Naive Bayes accuracy:     {accuracy:.2%}")

# STEP 7: SAVE MODEL AND RESULTS

print("\n[6/6] Saving model and results...")

joblib.dump(
    model,
    MODEL_DIR / "naive_bayes_model.joblib"
)

# Save thresholds for use in the website
# and consistent prediction interpretation.
threshold_records = thresholds.reset_index()
threshold_records.to_csv(
    MODEL_DIR / "traffic_thresholds.csv",
    index=False
)

with open(
    MODEL_DIR / "traffic_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "global_low": float(global_low),
        "global_high": float(global_high),
        "classes": labels,
        "method": "Per-restaurant training terciles"
    }, file, indent=2)

results = test[
    ["air_store_id", "visit_date", "visitors", "traffic_level"]
].copy()

results["predicted_traffic"] = predictions

results.to_csv(
    RESULT_DIR / "naive_bayes_results.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "Gaussian Naive Bayes",
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "baseline_accuracy": baseline_accuracy,
    "training_rows": len(train),
    "testing_rows": len(test)
}])

metrics.to_csv(
    RESULT_DIR / "naive_bayes_metrics.csv",
    index=False
)

# SAMPLE PREDICTION

sample = X_test.iloc[[0]]
sample_prediction = model.predict(sample)[0]
sample_probabilities = model.predict_proba(sample)[0]

print("\nSAMPLE PREDICTION")
print("-" * 40)
print("Restaurant:", test.iloc[0]["air_store_id"])
print("Predicted traffic:", sample_prediction)
print("Actual traffic:", y_test.iloc[0])

print("\nModel class probabilities:")
for name, probability in zip(
    model.named_steps["classifier"].classes_,
    sample_probabilities
):
    print(f"{name}: {probability:.2%}")

print("\n" + "=" * 55)
print("ML-02 NAIVE BAYES COMPLETED!")
print("=" * 55)
