
# ============================================
# DINEMIND AI JAPAN 2.0
# FASTAPI BACKEND
# ============================================

from pathlib import Path
import sys

from fastapi import FastAPI

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

from ml.prediction_service import predict_sentiment

app = FastAPI(
    title="DineMind AI API",
    description="Machine Learning Restaurant Analytics",
    version="2.0.0"
)


@app.get("/")
def home():
    return {
        "application": "DineMind AI Japan",
        "status": "Backend is running",
        "version": "2.0"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "FastAPI is operational"
    }


@app.get("/demo/sentiment")
def sentiment_demo():
    review = "The food was delicious and the staff were friendly."
    return predict_sentiment(review)
