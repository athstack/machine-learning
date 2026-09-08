# ============================================================
# QUESTION 19: DEPLOYMENT API
# ============================================================

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Retail Customer Churn Prediction API",
    description="Machine Learning API for customer churn prediction",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load(
    "../models/final_churn_model.joblib"
)


# ------------------------------------------------------------
# FORM PAGE
# ------------------------------------------------------------

@app.get("/")
def get_form():
    form_path = os.path.join(os.path.dirname(__file__), "form.html")
    return FileResponse(form_path, media_type="text/html")


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "customer churn prediction model"
    }


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

@app.post("/predict")
def predict(customer_data: dict):

    # Convert input data into DataFrame
    input_data = pd.DataFrame(
        [customer_data]
    )

    # Generate prediction
    prediction = int(
        model.predict(input_data)[0]
    )

    # Generate churn probability
    probability = float(
        model.predict_proba(
            input_data
        )[0][1]
    )

    # Determine risk level
    if probability >= 0.70:

        risk_level = "High Risk"

    elif probability >= 0.40:

        risk_level = "Medium Risk"

    else:

        risk_level = "Low Risk"

    return {
        "prediction": prediction,
        "churn_probability": round(
            probability,
            4
        ),
        "risk_level": risk_level
    }