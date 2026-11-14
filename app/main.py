from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartData

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

app = FastAPI(
    title="Heart Disease Prediction API",
    description="A simple student project that demonstrates a machine-learning API.",
    version="1.0.0",
)

model = None
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Welcome! The Heart Disease Prediction API is running.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.get("/info")
def info():
    return {
        "model_type": "RandomForestClassifier",
        "features": FEATURES,
        "target": {"0": "No heart disease", "1": "Heart disease"},
        "note": "This model is for educational use only."
    }


@app.post("/predict")
def predict(data: HeartData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The model file is missing. Run train_model.py first."
        )

    values = data.model_dump()
    input_data = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(input_data)[0])

    return {
        "heart_disease": bool(prediction),
        "prediction": prediction,
        "message": (
            "The model predicts heart disease."
            if prediction == 1
            else "The model predicts no heart disease."
        ),
        "disclaimer": "This is not a medical diagnosis. Please consult a qualified doctor."
    }
