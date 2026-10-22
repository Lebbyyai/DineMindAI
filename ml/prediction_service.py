
# ==============================================
# DINEMIND AI JAPAN 2.0
# UNIFIED MACHINE LEARNING PREDICTION SERVICE
# SIX DIFFERENT BUSINESS APPLICATIONS
# ==============================================

from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# Reuse our tested historical feature builder.
from ml.prediction_service_v1 import (
    build_features as build_traffic_features
)

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
RAW_DIR = BASE / "datasets" / "raw"

MODEL_FILES = {
    "revenue": "linear_regression_revenue_model.joblib",
    "sentiment": "naive_bayes_sentiment_model.joblib",
    "recommendation": "knn_recommender_model.joblib",
    "satisfaction": "svm_satisfaction_model.joblib",
    "traffic": "decision_tree_model.joblib",
    "visitors": "ann_model.joblib",
}

NUMERIC_FEATURES = [
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


# ----------------------------------------------
# LOAD TRAINED MODELS
# ----------------------------------------------

@lru_cache(maxsize=6)
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    path = MODEL_DIR / MODEL_FILES[model_name]

    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    # Only load models produced by our own
    # trusted Python training scripts.
    return joblib.load(path)


# ----------------------------------------------
# ML-01: LINEAR REGRESSION
# RESTAURANT REVENUE
# ----------------------------------------------

def predict_revenue(restaurant_data):
    """
    restaurant_data must contain:
    opening_year, opening_month,
    P1-P5, City Group, Type.
    """

    model = load_model("revenue")
    features = list(model.feature_names_in_)

    data = pd.DataFrame([restaurant_data])
    missing = [c for c in features if c not in data]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    prediction = float(model.predict(data[features])[0])

    return {
        "algorithm": "Linear Regression",
        "application": "Restaurant Revenue Prediction",
        "predicted_revenue": round(max(0, prediction), 2),
        "unit": "Dataset revenue units",
        "note": "The dataset is anonymized; results are estimates."
    }


# ----------------------------------------------
# ML-02: NAIVE BAYES
# CUSTOMER REVIEW SENTIMENT
# ----------------------------------------------

def predict_sentiment(review):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Enter a customer review.")

    model = load_model("sentiment")

    prediction = int(model.predict([review])[0])
    probabilities = model.predict_proba([review])[0]

    probability_map = {
        int(label): float(prob)
        for label, prob in zip(
            model.classes_, probabilities
        )
    }

    return {
        "algorithm": "Naive Bayes",
        "application": "Customer Review Sentiment",
        "review": review,
        "sentiment": (
            "Positive" if prediction == 1 else "Negative"
        ),
        "probabilities": {
            "positive": round(
                probability_map.get(1, 0), 4
            ),
            "negative": round(
                probability_map.get(0, 0), 4
            )
        },
        "note": "Trained on English-language Yelp reviews."
    }


# ----------------------------------------------
# ML-03: KNN
# RESTAURANT RECOMMENDATIONS
# ----------------------------------------------

def recommend_restaurants(
    price="medium",
    ambience="familiar",
    cuisine="Mexican",
    top_n=5
):
    bundle = load_model("recommendation")

    model = bundle["model"]
    restaurants = bundle["restaurants"]
    columns = bundle["feature_columns"]
    weights = np.asarray(bundle["weights"])

    query = pd.Series(0.0, index=columns)

    selected = [
        f"price_{price}",
        f"ambience_{ambience}",
    ]

    if cuisine:
        selected.append(f"cuisine_{cuisine}")

    for col in selected:
        if col not in query.index:
            raise ValueError(
                f"Preference unavailable in dataset: {col}"
            )
        query[col] = 1.0

    vector = (
        query.to_numpy(dtype=float) * weights
    ).reshape(1, -1)

    count = max(1, min(int(top_n), 20, len(restaurants)))

    distances, indices = model.kneighbors(
        vector,
        n_neighbors=count
    )

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        place = restaurants.iloc[int(idx)]

        rating = place["average_rating"]

        results.append({
            "restaurant_id": int(place["placeID"]),
            "name": str(place["name"]),
            "price": str(place["price"]),
            "ambience": str(place["Rambience"]),
            "cuisines": list(place["cuisines"]),
            "average_rating": (
                None if pd.isna(rating)
                else round(float(rating), 2)
            ),
            "rating_count": int(place["rating_count"]),
            "similarity": round(1 - float(distance), 4)
        })

    return {
        "algorithm": "K-Nearest Neighbors",
        "application": "Restaurant Recommendations",
        "recommendations": results,
        "note": (
            "Similarity measures attribute matching, "
            "not customer satisfaction."
        )
    }


# ----------------------------------------------
# ML-04: SVM
# CUSTOMER SATISFACTION
# ----------------------------------------------

def predict_satisfaction(user_id, restaurant_id):
    bundle = load_model("satisfaction")

    model = bundle["model"]
    history = bundle["historical_ratings"]
    places = bundle["restaurant_attributes"]
    features = bundle["features"]

    place = places.loc[
        places["placeID"] == restaurant_id
    ]

    if place.empty:
        raise ValueError("Restaurant ID not found.")

    global_mean = float(history["satisfied"].mean())

    def smoothed_mean(records):
        return (
            records["satisfied"].sum() + 5 * global_mean
        ) / (len(records) + 5)

    user_history = history.loc[
        history["userID"] == user_id
    ]

    restaurant_history = history.loc[
        history["placeID"] == restaurant_id
    ]

    row = place.iloc[0]

    input_data = pd.DataFrame([{
        "price": str(row["price"]),
        "Rambience": str(row["Rambience"]),
        "cuisines": str(row["cuisines"]),
        "user_history": smoothed_mean(user_history),
        "restaurant_history": smoothed_mean(
            restaurant_history
        ),
    }])

    predicted = int(model.predict(input_data[features])[0])

    return {
        "algorithm": "Support Vector Machine",
        "application": "Customer Satisfaction Prediction",
        "user_id": str(user_id),
        "restaurant_id": int(restaurant_id),
        "prediction": (
            "Satisfied" if predicted == 1
            else "Not Satisfied"
        ),
        "note": (
            "An experimental prediction derived from "
            "historical customer ratings."
        )
    }


# ----------------------------------------------
# ML-05: DECISION TREE
# RESTAURANT TRAFFIC CLASSIFICATION
# ----------------------------------------------

def predict_traffic(restaurant_id, date):
    model = load_model("traffic")

    row, actual = build_traffic_features(
        restaurant_id, date
    )

    features = NUMERIC_FEATURES + [
        "low_limit",
        "high_limit"
    ]

    prediction = str(model.predict(row[features])[0])

    return {
        "algorithm": "Decision Tree",
        "application": "Restaurant Traffic Classification",
        "restaurant_id": restaurant_id,
        "date": str(pd.Timestamp(date).date()),
        "predicted_traffic": prediction,
        "actual_visitors": int(actual),
        "note": "Historical rolling prediction, 2017."
    }


# ----------------------------------------------
# ML-06: ANN
# RESTAURANT VISITOR FORECAST
# ----------------------------------------------

def predict_visitors(restaurant_id, date):
    model = load_model("visitors")

    row, actual = build_traffic_features(
        restaurant_id, date
    )

    predicted = float(
        model.predict(row[NUMERIC_FEATURES])[0]
    )

    count = max(0, int(round(predicted)))

    return {
        "algorithm": "Artificial Neural Network",
        "application": "Restaurant Visitor Forecasting",
        "restaurant_id": restaurant_id,
        "date": str(pd.Timestamp(date).date()),
        "predicted_visitors": count,
        "actual_visitors": int(actual),
        "absolute_error": abs(count - int(actual)),
        "note": "Historical rolling prediction, 2017."
    }


# ----------------------------------------------
# TEST ALL SIX APPLICATIONS
# ----------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 55)
    print("DINEMIND AI 2.0 - SIX MODEL SERVICE TEST")
    print("=" * 55)

    # Revenue example: use real dataset attributes.
    revenue_data = pd.read_csv(
        RAW_DIR / "revenue" / "train.csv"
    )
    sample_revenue = revenue_data.iloc[0]
    opened = pd.to_datetime(
        sample_revenue["Open Date"],
        format="%m/%d/%Y"
    )

    revenue_input = {
        "opening_year": opened.year,
        "opening_month": opened.month,
        "City Group": sample_revenue["City Group"],
        "Type": sample_revenue["Type"],
        **{
            f"P{i}": float(sample_revenue[f"P{i}"])
            for i in range(1, 6)
        }
    }

    # Satisfaction example: use identifiers that
    # exist in the original ratings dataset.
    ratings = pd.read_csv(
        RAW_DIR / "recommendation" / "rating_final.csv",
        encoding="latin1"
    )
    sample_rating = ratings.iloc[0]

    # Historical Japanese restaurant example.
    restaurant_id = "air_00a91d42b08b08d9"
    date = "2017-03-15"

    checks = [
        ("Revenue", lambda: predict_revenue(revenue_input)),
        ("Sentiment", lambda: predict_sentiment(
            "The food was delicious and the staff were friendly."
        )),
        ("Recommendations", lambda: recommend_restaurants()),
        ("Satisfaction", lambda: predict_satisfaction(
            sample_rating["userID"],
            int(sample_rating["placeID"])
        )),
        ("Traffic", lambda: predict_traffic(
            restaurant_id, date
        )),
        ("Visitors", lambda: predict_visitors(
            restaurant_id, date
        )),
    ]

    failures = []

    for name, test in checks:
        print(f"\n--- {name.upper()} ---", flush=True)

        try:
            result = test()
            print(json.dumps(result, indent=2))
            print("PASS", flush=True)
        except Exception as error:
            failures.append((name, str(error)))
            print(f"FAIL: {error}", flush=True)

    print("\n" + "=" * 55)
    print(f"PASSED: {len(checks) - len(failures)}/6")

    if failures:
        print("Failed modules:")
        for name, error in failures:
            print(f"- {name}: {error}")
    else:
        print("ALL SIX APPLICATIONS TESTED SUCCESSFULLY!")
