
# ============================================
# DINEMIND AI JAPAN 2.0
# COMPLETE FASTAPI BACKEND
# ============================================

from datetime import date
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.prediction_service import (
    predict_revenue,
    predict_sentiment,
    recommend_restaurants,
    predict_satisfaction,
    predict_traffic,
    predict_visitors,
)

from ml.prediction_service_v1 import get_restaurants

# Create the API
app = FastAPI(
    title="DineMind AI Japan",
    description="Six Machine Learning Restaurant Applications",
    version="2.0.0"
)

# Allow our future local React website to use the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================
# REQUEST MODELS
# ============================================

class SentimentRequest(BaseModel):
    review: str = Field(min_length=1, max_length=3000)


class RevenueRequest(BaseModel):
    opening_year: int = Field(ge=1900, le=2026)
    opening_month: int = Field(ge=1, le=12)
    city_group: str
    restaurant_type: str
    P1: float
    P2: float
    P3: float
    P4: float
    P5: float


class SatisfactionRequest(BaseModel):
    user_id: str
    restaurant_id: int


class HistoricalPredictionRequest(BaseModel):
    restaurant_id: str
    prediction_date: date


# ============================================
# ERROR HANDLING
# ============================================

def execute_prediction(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)

    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="A required model or dataset is unavailable."
        ) from error


# ============================================
# BASIC ENDPOINTS
# ============================================

@app.get("/")
def home():
    return {
        "application": "DineMind AI Japan",
        "version": "2.0",
        "status": "Backend is running",
        "available_algorithms": 6
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "FastAPI server is operational"
    }


# ============================================
# ML-01: LINEAR REGRESSION
# ============================================

@app.post("/predict/revenue")
def revenue_endpoint(request: RevenueRequest):

    values = request.model_dump()

    model_input = {
        "opening_year": values["opening_year"],
        "opening_month": values["opening_month"],
        "City Group": values["city_group"],
        "Type": values["restaurant_type"],
        **{
            f"P{i}": values[f"P{i}"]
            for i in range(1, 6)
        }
    }

    return execute_prediction(
        predict_revenue,
        model_input
    )


# ============================================
# ML-02: NAIVE BAYES
# ============================================

@app.post("/predict/sentiment")
def sentiment_endpoint(request: SentimentRequest):

    return execute_prediction(
        predict_sentiment,
        request.review
    )


# ============================================
# ML-03: KNN
# ============================================

@app.get("/predict/recommendations")
def recommendation_endpoint(
    price: str = "medium",
    ambience: str = "familiar",
    cuisine: str = "Mexican",
    top_n: int = Query(default=5, ge=1, le=20)
):

    return execute_prediction(
        recommend_restaurants,
        price=price,
        ambience=ambience,
        cuisine=cuisine,
        top_n=top_n
    )


# ============================================
# ML-04: SVM
# ============================================

@app.post("/predict/satisfaction")
def satisfaction_endpoint(
    request: SatisfactionRequest
):

    return execute_prediction(
        predict_satisfaction,
        request.user_id,
        request.restaurant_id
    )


# ============================================
# ML-05: DECISION TREE
# ============================================

@app.post("/predict/traffic")
def traffic_endpoint(
    request: HistoricalPredictionRequest
):

    return execute_prediction(
        predict_traffic,
        request.restaurant_id,
        request.prediction_date.isoformat()
    )


# ============================================
# ML-06: ARTIFICIAL NEURAL NETWORK
# ============================================

@app.post("/predict/visitors")
def visitors_endpoint(
    request: HistoricalPredictionRequest
):

    return execute_prediction(
        predict_visitors,
        request.restaurant_id,
        request.prediction_date.isoformat()
    )


# ============================================
# AVAILABLE RESTAURANTS
# ============================================

@app.get("/restaurants/japan")
def japanese_restaurants():

    return execute_prediction(get_restaurants)


# ============================================
# KNN: AVAILABLE RESTAURANT PREFERENCES
# ============================================

from ml.prediction_service import load_model

@app.get("/restaurants/preferences")
def restaurant_preferences():
    bundle = load_model("recommendation")
    restaurants = bundle["restaurants"]

    prices = sorted(
        restaurants["price"].dropna().unique().tolist()
    )

    ambiences = sorted(
        restaurants["Rambience"].dropna().unique().tolist()
    )

    cuisines = bundle["cuisines"]

    return {
        "prices": prices,
        "ambiences": ambiences,
        "cuisines": cuisines,
    }


# ============================================
# SVM CUSTOMER SATISFACTION - FORM OPTIONS
# ============================================

from ml.prediction_service import load_model


@app.get("/satisfaction/options")
def satisfaction_options():
    bundle = load_model("satisfaction")

    history = bundle["historical_ratings"]
    places = bundle["restaurant_attributes"].copy()

    customers = sorted(
        history["userID"].dropna().astype(str).unique().tolist()
    )

    # Add readable restaurant names from geoplaces2.csv
    import pandas as pd

    names = pd.read_csv(
        BASE_DIR / "datasets" / "raw" /
        "recommendation" / "geoplaces2.csv",
        encoding="latin1",
        usecols=["placeID", "name"]
    )

    places = places.merge(
        names,
        on="placeID",
        how="left",
        validate="one_to_one"
    )

    restaurants = (
        places[["placeID", "name"]]
        .sort_values("name")
        .to_dict(orient="records")
    )

    return {
        "customers": customers,
        "restaurants": restaurants
    }
