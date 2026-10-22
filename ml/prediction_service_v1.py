
# ==============================================
# DINEMIND AI JAPAN
# PREDICTION SERVICE - ALL SIX ML MODELS
# ==============================================

from pathlib import Path
from functools import lru_cache
import json

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
MODEL_DIR = BASE_DIR / "models"

TEST_START = pd.Timestamp("2017-03-01")
TEST_END = pd.Timestamp("2017-04-22")

MODEL_FILES = {
    "linear_regression": "linear_regression_model.joblib",
    "naive_bayes": "naive_bayes_model.joblib",
    "knn": "knn_model.joblib",
    "svm": "svm_model.joblib",
    "decision_tree": "decision_tree_model.joblib",
    "ann": "ann_model.joblib",
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

FEATURE_SETS = {
    "linear_regression": NUMERIC_FEATURES + [
        "air_store_id",
        "air_genre_name",
        "air_area_name",
    ],
    "naive_bayes": NUMERIC_FEATURES + [
        "air_store_id",
        "air_genre_name",
    ],
    "knn": [
        "month",
        "day_of_week_num",
        "is_weekend",
        "holiday_flg",
        "previous_visitors",
        "previous_7_record_avg",
        "previous_14_record_avg",
        "days_since_previous_record",
    ],
    "svm": NUMERIC_FEATURES,
    "decision_tree": NUMERIC_FEATURES + [
        "low_limit",
        "high_limit",
    ],
    "ann": NUMERIC_FEATURES,
}


@lru_cache(maxsize=1)
def load_data():
    visits = pd.read_csv(
        RAW_DIR / "air_visit_data.csv",
        parse_dates=["visit_date"]
    )
    stores = pd.read_csv(
        RAW_DIR / "air_store_info.csv"
    )
    calendar = pd.read_csv(
        RAW_DIR / "date_info.csv",
        parse_dates=["calendar_date"]
    )

    thresholds = pd.read_csv(
        MODEL_DIR / "traffic_thresholds.csv"
    )

    with open(
        MODEL_DIR / "traffic_config.json",
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    visits = visits.sort_values(
        ["air_store_id", "visit_date"]
    )

    return visits, stores, calendar, thresholds, config


@lru_cache(maxsize=6)
def load_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    path = MODEL_DIR / MODEL_FILES[model_name]

    if not path.exists():
        raise FileNotFoundError(
            f"Model missing: {path}"
        )

    # Only load files created by our trusted
    # model-training scripts.
    return joblib.load(path)


def get_restaurants():
    """Return restaurants available in the test period."""
    visits, stores, _, _, _ = load_data()

    eligible_ids = visits.loc[
        (visits["visit_date"] >= TEST_START) &
        (visits["visit_date"] <= TEST_END),
        "air_store_id"
    ].unique()

    result = stores[
        stores["air_store_id"].isin(eligible_ids)
    ][[
        "air_store_id",
        "air_genre_name",
        "air_area_name"
    ]].copy()

    return result.sort_values(
        "air_store_id"
    ).to_dict(orient="records")


def build_features(restaurant_id, prediction_date):
    """
    Prepare features for a historical, one-day-ahead
    prediction using recorded visits BEFORE the date.

    The target day's visitor count is never used
    to construct prediction features.
    """

    visits, stores, calendar, thresholds, config = load_data()

    date = pd.Timestamp(prediction_date).normalize()

    if date < TEST_START or date > TEST_END:
        raise ValueError(
            "Choose a historical date from "
            "2017-03-01 through 2017-04-22."
        )

    store_info = stores[
        stores["air_store_id"] == restaurant_id
    ]

    if store_info.empty:
        raise ValueError("Restaurant ID not found.")

    # Historical demonstration only:
    # Require a recorded observation on the
    # selected date so results can be evaluated.
    current_record = visits[
        (visits["air_store_id"] == restaurant_id) &
        (visits["visit_date"] == date)
    ]

    if current_record.empty:
        raise ValueError(
            "No recorded visit count for this restaurant "
            "on the selected date. Choose another date."
        )

    # Filter by date BEFORE calculating history.
    history = visits[
        (visits["air_store_id"] == restaurant_id) &
        (visits["visit_date"] < date)
    ].sort_values("visit_date")

    if len(history) < 14:
        raise ValueError(
            "At least 14 earlier visit records are required."
        )

    calendar_info = calendar[
        calendar["calendar_date"] == date
    ]

    if calendar_info.empty:
        raise ValueError(
            "Calendar information is unavailable."
        )

    store = store_info.iloc[0]

    threshold_row = thresholds[
        thresholds["air_store_id"] == restaurant_id
    ]

    if threshold_row.empty:
        low_limit = float(config["global_low"])
        high_limit = float(config["global_high"])
    else:
        low_limit = float(
            threshold_row.iloc[0]["low_limit"]
        )
        high_limit = float(
            threshold_row.iloc[0]["high_limit"]
        )

    previous_date = history.iloc[-1]["visit_date"]

    features = {
        "month": date.month,
        "day": date.day,
        "day_of_week_num": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "holiday_flg": int(
            calendar_info.iloc[0]["holiday_flg"]
        ),
        "previous_visitors": float(
            history.iloc[-1]["visitors"]
        ),
        "previous_7_record_avg": float(
            history["visitors"].tail(7).mean()
        ),
        "previous_14_record_avg": float(
            history["visitors"].tail(14).mean()
        ),
        "days_since_previous_record": int(
            (date - previous_date).days
        ),
        "air_store_id": restaurant_id,
        "air_genre_name": store["air_genre_name"],
        "air_area_name": store["air_area_name"],
        "low_limit": low_limit,
        "high_limit": high_limit,
    }

    return (
        pd.DataFrame([features]),
        int(current_record.iloc[0]["visitors"])
    )


def predict(restaurant_id, prediction_date, model_name):
    """
    Generate a prediction using one of
    the six trained machine learning models.
    """

    if model_name not in MODEL_FILES:
        raise ValueError(
            "Available models: " +
            ", ".join(MODEL_FILES.keys())
        )

    features, actual_visitors = build_features(
        restaurant_id,
        prediction_date
    )

    model = load_model(model_name)

    model_input = features[
        FEATURE_SETS[model_name]
    ]

    predicted = model.predict(model_input)[0]

    result = {
        "restaurant_id": restaurant_id,
        "prediction_date": str(
            pd.Timestamp(prediction_date).date()
        ),
        "model": model_name,
        "actual_visitors": actual_visitors,
        "prediction_type": "historical_rolling",
    }

    if model_name in ("linear_regression", "ann"):
        prediction_count = max(
            0,
            int(round(float(predicted)))
        )

        result["predicted_visitors"] = prediction_count
        result["absolute_error"] = abs(
            actual_visitors - prediction_count
        )

    elif model_name == "svm":
        result["predicted_traffic"] = (
            "High" if int(predicted) == 1
            else "Not High"
        )

        high_limit = features.iloc[0]["high_limit"]

        result["actual_traffic"] = (
            "High" if actual_visitors > high_limit
            else "Not High"
        )

    else:
        result["predicted_traffic"] = str(predicted)

        low_limit = features.iloc[0]["low_limit"]
        high_limit = features.iloc[0]["high_limit"]

        if actual_visitors <= low_limit:
            actual_traffic = "Low"
        elif actual_visitors <= high_limit:
            actual_traffic = "Medium"
        else:
            actual_traffic = "High"

        result["actual_traffic"] = actual_traffic

    return result


# ==============================================
# LOCAL TEST - RUN FROM VS CODE TERMINAL
# ==============================================

if __name__ == "__main__":

    print("=" * 55)
    print("DINEMIND AI - PREDICTION SERVICE TEST")
    print("=" * 55)

    restaurants = get_restaurants()

    # Pick a restaurant with sufficient history
    # and an observation on the example date.
    example_date = "2017-03-15"
    selected_id = None

    for restaurant in restaurants:
        try:
            build_features(
                restaurant["air_store_id"],
                example_date
            )
            selected_id = restaurant["air_store_id"]
            break
        except ValueError:
            continue

    if selected_id is None:
        raise RuntimeError(
            "No eligible restaurant found for test date."
        )

    print("\nRestaurant:", selected_id)
    print("Date:", example_date)

    for name in MODEL_FILES:
        print(f"\n--- {name.upper()} ---")

        result = predict(
            selected_id,
            example_date,
            name
        )

        print(json.dumps(result, indent=2))

    print("\n" + "=" * 55)
    print("ALL SIX PREDICTION TESTS COMPLETED!")
    print("=" * 55)
