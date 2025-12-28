from fastapi import FastAPI #API framework
from pydantic import BaseModel, Field
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Load trained CSAT model
model = joblib.load("../models/csat_model.pkl")
analyzer = SentimentIntensityAnalyzer()

# Topic -> Recommendation mapping
recommendations = {
    0: "Improve flight scheduling and delay communication",
    1: "Enhance seat comfort and legroom",
    2: "Improve in-flight food quality and options",
    3: "Provide better customer service and staff training",
    4: "Improve baggage handling and tracking systems"
}

class ReviewInput(BaseModel):
    review: str
    GroundService: float = Field(default=3.0, ge=1, le=5, description="Ground service rating (1-5)")
    FoodBeverages: float = Field(default=3.0, ge=1, le=5, description="Food & beverages rating (1-5)")
    ValueForMoney: float = Field(default=3.0, ge=1, le=5, description="Value for money rating (1-5)")

@app.get("/")
def root():
    return {
        "message": "Customer Sentiment Prediction API",
        "endpoints": {
            "/predict": "POST - Predict customer satisfaction from review text",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.post("/predict")
def predict(data: ReviewInput):
    # Sentiment score
    sentiment = analyzer.polarity_scores(data.review)["compound"]

    # Dummy topic ID for recommendation (simple hash)
    topic_id = abs(hash(data.review)) % 5
    recommendation = recommendations[topic_id]

    # Create feature vector with all 4 features: sentiment_score, GroundService, Food&Beverages, ValueForMoney
    features = np.array([[sentiment, data.GroundService, data.FoodBeverages, data.ValueForMoney]])

    # Predict CSAT
    csat_prediction = model.predict(features)[0]

    return {
        "sentiment_score": round(float(sentiment), 3),
        "predicted_csat": round(float(csat_prediction), 2),
        "recommendation": recommendation,
        "inputs_used": {
            "GroundService": data.GroundService,
            "FoodBeverages": data.FoodBeverages,
            "ValueForMoney": data.ValueForMoney
        }
    }
