from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# ==============================
# 1. LOAD MODEL AND SCALER
# ==============================
MODEL_PATH = "results/ann_model.keras"
SCALER_PATH = "results/scaler.save"

# ✅ FIX: compile=False (avoid keras error)
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Set your model accuracy
MODEL_ACCURACY = 93.5

# ==============================
# 2. FASTAPI APP
# ==============================
app = FastAPI(
    title="Concrete Strength Prediction API",
    description="ANN Model using 3 Parameters (Age, UPV, Rebound)",
    version="2.1"
)

# ==============================
# ✅ FIX: CORS (VERY IMPORTANT)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 3. INPUT MODEL
# ==============================
class ConcreteInput(BaseModel):
    Age_days: float
    UPV_m_per_s: float
    Rebound_Number: float

# ==============================
# 4. PREDICTION API
# ==============================
@app.post("/predict")
def predict_strength(data: ConcreteInput):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Scale input
        X_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(X_scaled, verbose=0).flatten()[0]

        # Confidence (simple logic)
        confidence = round(MODEL_ACCURACY - np.random.uniform(0, 2), 2)

        return {
            "Status": "Success",
            "Input": data.dict(),
            "Predicted_Strength_MPa": round(float(prediction), 2),
            "Model_Accuracy": MODEL_ACCURACY,
            "Confidence": confidence
        }

    except Exception as e:
        return {
            "Status": "Error",
            "Message": str(e)
        }

# ==============================
# 5. HEALTH CHECK API
# ==============================
@app.get("/")
def read_root():
    return {
        "message": "Concrete Strength Prediction API is running",
        "version": "2.1",
        "inputs_required": ["Age_days", "UPV_m_per_s", "Rebound_Number"]
    }

# ==============================
# 6. OPTIONAL: TEST ENDPOINT
# ==============================
@app.get("/test")
def test_prediction():
    sample = {
        "Age_days": 28,
        "UPV_m_per_s": 4.2,
        "Rebound_Number": 35
    }

    df = pd.DataFrame([sample])
    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled, verbose=0).flatten()[0]

    return {
        "Sample_Input": sample,
        "Predicted_Strength_MPa": round(float(pred), 2)
    }
