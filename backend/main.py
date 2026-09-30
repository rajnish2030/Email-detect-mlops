from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os


MODEL_PATH = "backend/models/spam_classifier.pkl"


app = FastAPI(
    title="Email Spam Detection API",
    description="Machine Learning API for Email Spam Detection",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run train.py first."
    )


model = joblib.load(MODEL_PATH)


class EmailRequest(BaseModel):
    email: str


@app.get("/")
def home():
    return {
        "message": "Email Spam Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(request: EmailRequest):

    email = request.email.strip()

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email text cannot be empty"
        )

    prediction = model.predict([email])[0]

    probabilities = model.predict_proba([email])[0]

    confidence = float(
        max(probabilities) * 100
    )

    label = (
        "SPAM"
        if prediction == 1
        else "NOT SPAM"
    )

    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }

