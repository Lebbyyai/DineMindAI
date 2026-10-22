
# =============================================
# DINEMIND AI 2.0
# ML-04: SVM CUSTOMER SATISFACTION PREDICTION
# =============================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "datasets" / "raw" / "recommendation"
MODELS = BASE / "models"
RESULTS = BASE / "results"

MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("=" * 55)
print("DINEMIND AI - SVM CUSTOMER SATISFACTION")
print("=" * 55)

# 1. LOAD DATA
print("\n[1/7] Loading data...", flush=True)

ratings = pd.read_csv(
    DATA / "rating_final.csv", encoding="latin1"
)
places = pd.read_csv(
    DATA / "geoplaces2.csv", encoding="latin1"
)
cuisines = pd.read_csv(
    DATA / "chefmozcuisine.csv", encoding="latin1"
)

# 2. PREPARE RESTAURANT ATTRIBUTES
print("\n[2/7] Preparing restaurant data...")

cuisine_info = (
    cuisines.groupby("placeID")["Rcuisine"]
    .apply(lambda x: "|".join(sorted(set(x.dropna()))))
    .reset_index(name="cuisines")
)

place_info = places[
    ["placeID", "price", "Rambience"]
].merge(
    cuisine_info, on="placeID", how="left",
    validate="one_to_one"
)

data = ratings.merge(
    place_info, on="placeID", how="left",
    validate="many_to_one"
)

for col in ["price", "Rambience", "cuisines"]:
    data[col] = data[col].fillna("unknown").astype(str)

# A rating of 2 represents satisfied.
# Ratings 0 and 1 represent not satisfied.
data["satisfied"] = (data["rating"] == 2).astype(int)

print("Rating records:", len(data))
print("Class distribution:")
print(data["satisfied"].value_counts())

# 3. TRAIN / TEST SPLIT
print("\n[3/7] Splitting data...")

train, test = train_test_split(
    data, test_size=0.20,
    stratify=data["satisfied"],
    random_state=42
)

fit, validation = train_test_split(
    train, test_size=0.25,
    stratify=train["satisfied"],
    random_state=42
)

# 4. CREATE HISTORICAL RATING FEATURES
# Training rows use leave-one-out means.
# Validation/test use ONLY the available
# earlier training subset.

def add_history(reference, target, leave_one_out=False):
    result = target.copy()

    global_mean = reference["satisfied"].mean()

    for key, prefix in [
        ("userID", "user"),
        ("placeID", "restaurant")
    ]:
        stats = reference.groupby(key)["satisfied"].agg(
            ["sum", "count"]
        )

        sums = result[key].map(stats["sum"]).fillna(0)
        counts = result[key].map(stats["count"]).fillna(0)

        if leave_one_out:
            sums = sums - result["satisfied"]
            counts = counts - 1

        # Smooth toward overall training mean,
        # especially for customers with few ratings.
        result[f"{prefix}_history"] = (
            (sums + 5 * global_mean) / (counts + 5)
        )

    return result

categorical = ["price", "Rambience", "cuisines"]
numerical = ["user_history", "restaurant_history"]
features = categorical + numerical

def build_model(c):
    preprocessing = ColumnTransformer([
        ("categories", OneHotEncoder(
            handle_unknown="ignore"
        ), categorical),
        ("numbers", StandardScaler(), numerical)
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("svm", SVC(
            kernel="rbf",
            C=c,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

# 5. SELECT SVM PARAMETERS
print("\n[4/7] Selecting SVM settings...", flush=True)

fit_features = add_history(fit, fit, True)
validation_features = add_history(fit, validation)

best_c = None
best_f1 = -1

for c in [0.1, 1, 10]:
    candidate = build_model(c)
    candidate.fit(
        fit_features[features],
        fit_features["satisfied"]
    )

    predictions = candidate.predict(
        validation_features[features]
    )

    score = f1_score(
        validation_features["satisfied"],
        predictions,
        zero_division=0
    )

    print(f"C={c}: Validation F1={score:.4f}", flush=True)

    if score > best_f1:
        best_f1 = score
        best_c = c

print("Selected C:", best_c)

# 6. FINAL TRAINING AND EVALUATION
print("\n[5/7] Training final SVM...", flush=True)

train_features = add_history(train, train, True)
test_features = add_history(train, test)

model = build_model(best_c)
model.fit(
    train_features[features],
    train_features["satisfied"]
)

print("\n[6/7] Evaluating...", flush=True)

actual = test_features["satisfied"]
predicted = model.predict(test_features[features])

accuracy = accuracy_score(actual, predicted)
precision = precision_score(
    actual, predicted, zero_division=0
)
recall = recall_score(
    actual, predicted, zero_division=0
)
f1 = f1_score(actual, predicted, zero_division=0)

majority_class = int(train["satisfied"].mean() >= 0.5)
baseline = np.full(len(test), majority_class)
baseline_accuracy = accuracy_score(actual, baseline)

print("\n" + "=" * 55)
print("SVM SATISFACTION RESULTS")
print("=" * 55)
print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Majority baseline accuracy: {baseline_accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    actual, predicted,
    target_names=["Not Satisfied", "Satisfied"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

# 7. SAVE MODEL AND SUPPORTING DATA
print("\n[7/7] Saving model...", flush=True)

bundle = {
    "model": model,
    "features": features,
    "historical_ratings": train[
        ["userID", "placeID", "satisfied"]
    ].copy(),
    "restaurant_attributes": place_info,
    "selected_c": best_c,
}

joblib.dump(
    bundle,
    MODELS / "svm_satisfaction_model.joblib"
)

pd.DataFrame([{
    "model": "SVM Customer Satisfaction",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "validation_f1": best_f1,
    "baseline_accuracy": baseline_accuracy,
    "train_rows": len(train),
    "test_rows": len(test)
}]).to_csv(
    RESULTS / "svm_satisfaction_metrics.csv",
    index=False
)

output = test[["userID", "placeID", "rating"]].copy()
output["actual_satisfied"] = actual.to_numpy()
output["predicted_satisfied"] = predicted

output.to_csv(
    RESULTS / "svm_satisfaction_results.csv",
    index=False
)

print("\nML-04 SVM SATISFACTION MODEL COMPLETED!")
print("Saved: models/svm_satisfaction_model.joblib")
