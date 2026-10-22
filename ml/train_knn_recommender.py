
# ============================================
# DINEMIND AI JAPAN 2.0
# ML-03: KNN RESTAURANT RECOMMENDATION
# ============================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "datasets" / "raw" / "recommendation"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

print("=" * 55)
print("DINEMIND AI - KNN RESTAURANT RECOMMENDATION")
print("=" * 55)

# 1. LOAD DATASETS
print("\n[1/6] Loading datasets...")

places = pd.read_csv(
    DATA_DIR / "geoplaces2.csv",
    encoding="latin1"
)

cuisines = pd.read_csv(
    DATA_DIR / "chefmozcuisine.csv",
    encoding="latin1"
)

ratings = pd.read_csv(
    DATA_DIR / "rating_final.csv",
    encoding="latin1"
)

print("Restaurants:", len(places))
print("Cuisine records:", len(cuisines))
print("Customer ratings:", len(ratings))

# 2. PREPARE RESTAURANT PROFILES
print("\n[2/6] Preparing restaurant profiles...")

cuisine_groups = (
    cuisines.groupby("placeID")["Rcuisine"]
    .apply(lambda values: sorted(set(values.dropna())))
    .reset_index(name="cuisines")
)

restaurants = places.merge(
    cuisine_groups,
    on="placeID",
    how="left",
    validate="one_to_one"
)

restaurants["cuisines"] = restaurants["cuisines"].apply(
    lambda x: x if isinstance(x, list) else []
)

for col in ["price", "Rambience"]:
    restaurants[col] = (
        restaurants[col]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

# Aggregate historical ratings for display.
rating_summary = (
    ratings.groupby("placeID")
    .agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "size")
    )
    .reset_index()
)

restaurants = restaurants.merge(
    rating_summary,
    on="placeID",
    how="left",
    validate="one_to_one"
)

restaurants["rating_count"] = (
    restaurants["rating_count"].fillna(0).astype(int)
)

# 3. ENCODE RESTAURANT ATTRIBUTES
print("\n[3/6] Encoding restaurant characteristics...")

# Price and ambience encoded as one-hot vectors.
price_encoded = pd.get_dummies(
    restaurants["price"],
    prefix="price",
    dtype=float
)

ambience_encoded = pd.get_dummies(
    restaurants["Rambience"],
    prefix="ambience",
    dtype=float
)

# Multi-hot encode all cuisine types.
all_cuisines = sorted({
    cuisine
    for group in restaurants["cuisines"]
    for cuisine in group
})

cuisine_encoded = pd.DataFrame(
    {
        f"cuisine_{c}": restaurants["cuisines"].apply(
            lambda group: float(c in group)
        )
        for c in all_cuisines
    }
)

features = pd.concat(
    [
        price_encoded.reset_index(drop=True),
        ambience_encoded.reset_index(drop=True),
        cuisine_encoded.reset_index(drop=True)
    ],
    axis=1
)

# Give cuisine matches greater weight.
weights = np.array([
    3.0 if col.startswith("cuisine_") else 1.0
    for col in features.columns
])

feature_matrix = features.to_numpy(dtype=float) * weights

assert len(restaurants) == len(feature_matrix)

# 4. FIT NEAREST NEIGHBORS
print("\n[4/6] Fitting KNN recommender...")

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

model.fit(feature_matrix)

print("KNN model fitted!")
print("Feature count:", features.shape[1])

# 5. SAVE MODEL AND METADATA
print("\n[5/6] Saving model...")

bundle = {
    "model": model,
    "restaurants": restaurants,
    "feature_columns": features.columns.tolist(),
    "weights": weights,
    "cuisines": all_cuisines
}

joblib.dump(
    bundle,
    MODEL_DIR / "knn_recommender_model.joblib"
)

restaurants.to_csv(
    RESULT_DIR / "restaurant_profiles.csv",
    index=False
)

# 6. RECOMMENDATION FUNCTION

def recommend_restaurants(
    price="medium",
    ambience="familiar",
    cuisine=None,
    top_n=5
):
    """
    Find restaurants matching a customer's
    preferred price, ambience and cuisine.
    """

    query = pd.Series(
        0.0,
        index=features.columns
    )

    # User preferences
    for column in [
        f"price_{price}",
        f"ambience_{ambience}"
    ]:
        if column not in query.index:
            raise ValueError(
                f"Preference not found in dataset: {column}"
            )
        query[column] = 1.0

    if cuisine is not None:
        column = f"cuisine_{cuisine}"

        if column not in query.index:
            raise ValueError(
                f"Unknown cuisine: {cuisine}"
            )

        query[column] = 1.0

    query_vector = (
        query.to_numpy(dtype=float) * weights
    ).reshape(1, -1)

    distances, indices = model.kneighbors(
        query_vector,
        n_neighbors=min(top_n, len(restaurants))
    )

    matches = restaurants.iloc[
        indices[0]
    ].copy()

    matches["similarity"] = (
        1.0 - distances[0]
    )

    return matches[[
        "name",
        "price",
        "Rambience",
        "cuisines",
        "average_rating",
        "rating_count",
        "similarity"
    ]]


# SAMPLE RECOMMENDATION
print("\n[6/6] Testing restaurant recommendations...")

sample_cuisine = (
    "Mexican" if "Mexican" in all_cuisines
    else all_cuisines[0] if all_cuisines
    else None
)

print("\nSample preferences:")
print("Price: medium")
print("Ambience: familiar")
print("Cuisine:", sample_cuisine)

try:
    recommendations = recommend_restaurants(
        price="medium",
        ambience="familiar",
        cuisine=sample_cuisine,
        top_n=5
    )
except ValueError as error:
    print("Sample preference unavailable:", error)
    print("Testing available restaurant attributes instead.")

    first = restaurants.iloc[0]
    recommendations = recommend_restaurants(
        price=first["price"],
        ambience=first["Rambience"],
        cuisine=None,
        top_n=5
    )

print("\nTOP 5 RESTAURANT RECOMMENDATIONS")
print(recommendations.to_string(index=False))

print("\n" + "=" * 55)
print("ML-03 KNN RECOMMENDER COMPLETED!")
print("=" * 55)
print("Saved: models/knn_recommender_model.joblib")
