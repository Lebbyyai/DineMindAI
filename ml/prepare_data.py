
from pathlib import Path
import pandas as pd
import numpy as np

# =============================================
# DINEMIND AI JAPAN
# PHASE 2: RESTAURANT DATA PREPARATION
# =============================================

# Locate the project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "datasets" / "raw"
OUTPUT_DIR = BASE_DIR / "datasets" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("DINEMIND AI - RESTAURANT DATA PREPARATION")
print("=" * 55)

# STEP 1: LOAD DATASETS
print("\n[1/7] Loading restaurant datasets...")

visits = pd.read_csv(RAW_DIR / "air_visit_data.csv")
stores = pd.read_csv(RAW_DIR / "air_store_info.csv")
calendar = pd.read_csv(RAW_DIR / "date_info.csv")

print(f"Visitor records: {len(visits):,}")
print(f"Restaurants: {len(stores):,}")
print(f"Calendar records: {len(calendar):,}")

# STEP 2: VALIDATE AND CLEAN
print("\n[2/7] Validating data...")

visits["visit_date"] = pd.to_datetime(
    visits["visit_date"], errors="raise"
)
calendar["calendar_date"] = pd.to_datetime(
    calendar["calendar_date"], errors="raise"
)

assert not visits.duplicated(
    ["air_store_id", "visit_date"]
).any(), "Duplicate restaurant-date records found!"

assert not stores["air_store_id"].duplicated().any(), (
    "Duplicate restaurant IDs found!"
)

assert not calendar["calendar_date"].duplicated().any(), (
    "Duplicate calendar dates found!"
)

assert visits["visitors"].notna().all(), (
    "Missing visitor counts found!"
)

assert (visits["visitors"] >= 0).all(), (
    "Negative visitor counts found!"
)

# STEP 3: MERGE THE THREE DATASETS
print("\n[3/7] Creating master restaurant dataset...")

data = visits.merge(
    stores,
    on="air_store_id",
    how="left",
    validate="many_to_one"
)

data = data.merge(
    calendar,
    left_on="visit_date",
    right_on="calendar_date",
    how="left",
    validate="many_to_one"
)

assert len(data) == len(visits), (
    "Unexpected row count after merging!"
)

assert data["air_genre_name"].notna().all(), (
    "Some restaurant information is missing!"
)

assert data["holiday_flg"].notna().all(), (
    "Some calendar information is missing!"
)

data = data.drop(columns=["calendar_date"])

# STEP 4: CREATE CALENDAR FEATURES
print("\n[4/7] Engineering date features...")

data["year"] = data["visit_date"].dt.year
data["month"] = data["visit_date"].dt.month
data["day"] = data["visit_date"].dt.day
data["day_of_week_num"] = data["visit_date"].dt.dayofweek

data["is_weekend"] = (
    data["day_of_week_num"] >= 5
).astype(int)

data["holiday_flg"] = data["holiday_flg"].astype(int)

# STEP 5: CREATE HISTORICAL FEATURES
print("\n[5/7] Creating historical visitor features...")

data = data.sort_values(
    ["air_store_id", "visit_date"]
).reset_index(drop=True)

group = data.groupby("air_store_id")["visitors"]

# Shift ensures today's visitors are NEVER used
# to calculate today's input features.
data["previous_visitors"] = group.shift(1)

data["previous_7_record_avg"] = (
    data.groupby("air_store_id")["visitors"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=7,
            min_periods=7
        ).mean()
    )
)

data["previous_14_record_avg"] = (
    data.groupby("air_store_id")["visitors"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=14,
            min_periods=14
        ).mean()
    )
)

# Number of days since the last recorded visit
data["days_since_previous_record"] = (
    data.groupby("air_store_id")["visit_date"]
    .diff()
    .dt.days
)

# Keep rows with sufficient past observations.
# Do not replace unavailable history with zero.
model_data = data.dropna(
    subset=[
        "previous_visitors",
        "previous_7_record_avg",
        "previous_14_record_avg",
        "days_since_previous_record"
    ]
).copy()

# STEP 6: CHRONOLOGICAL TRAIN / TEST SPLIT
print("\n[6/7] Splitting training and testing data...")

# All training records precede testing records.
TEST_START = pd.Timestamp("2017-03-01")

train = model_data[
    model_data["visit_date"] < TEST_START
].copy()

test = model_data[
    model_data["visit_date"] >= TEST_START
].copy()

assert len(train) > 0, "Training set is empty!"
assert len(test) > 0, "Testing set is empty!"

assert train["visit_date"].max() < test["visit_date"].min()

# Sort chronologically for easier inspection
train = train.sort_values(
    ["visit_date", "air_store_id"]
)
test = test.sort_values(
    ["visit_date", "air_store_id"]
)

# STEP 7: SAVE OUTPUT FILES
print("\n[7/7] Saving prepared datasets...")

data.to_csv(
    OUTPUT_DIR / "master_restaurant_data.csv",
    index=False
)

train.to_csv(
    OUTPUT_DIR / "train_data.csv",
    index=False
)

test.to_csv(
    OUTPUT_DIR / "test_data.csv",
    index=False
)

print("\n" + "=" * 55)
print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
print("=" * 55)

print(f"Master dataset: {len(data):,} rows")
print(f"Model-ready records: {len(model_data):,} rows")
print(f"Training records: {len(train):,}")
print(f"Testing records: {len(test):,}")
print(f"Restaurants: {data['air_store_id'].nunique():,}")
print(f"Columns: {len(data.columns)}")

print("\nTraining period:")
print(train["visit_date"].min().date(), "to",
      train["visit_date"].max().date())

print("\nTesting period:")
print(test["visit_date"].min().date(), "to",
      test["visit_date"].max().date())

print("\nFiles saved to:", OUTPUT_DIR)
print("\nNEXT PHASE: LINEAR REGRESSION MODEL TRAINING")
