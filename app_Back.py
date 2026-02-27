from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Food Wastage Reduction API")

# =========================
# LOAD ML MODEL
# =========================
model = joblib.load("food_spoilage_model.pkl")

# =========================
# BASE FRESHNESS HOURS
# =========================
BASE_FRESHNESS = {
    "rice": 6,
    "chapathi": 6,
    "biryani": 8,
    "curry": 8,
    "fruit": 12,
    "packed": 24
}

# =========================
# BASE SERVING TABLE (Surplus Estimation)
# =========================
BASE_SERVING = {
    "rice": 0.25,      # kg per person
    "biryani": 0.35,   # kg per person
    "chapathi": 2,     # pieces per person
    "dosa": 2,         # pieces per person
    "idli": 3          # pieces per person
}

# =========================
# FRESHNESS DURATION LOGIC
# =========================
def freshness_duration(food, temp, humid, prepared_time):
    food = food.lower()

    # Default freshness if food not listed
    base = BASE_FRESHNESS.get(food, 6)

    reduction = 0
    if temp > 30:
        reduction += 0.30
    if humid > 90:
        reduction += 0.25

    adjusted_life = base * (1 - reduction)

    time_passed = (datetime.now() - prepared_time).total_seconds() / 3600
    remaining = adjusted_life - time_passed

    if remaining <= 0:
        return 0, "Spoiled"
    elif remaining < 1:
        return round(remaining, 2), "High Risk"
    else:
        return round(remaining, 2), "Safe"

# =========================
# INPUT SCHEMAS
# =========================
class SpoilageInput(BaseModel):
    temp: float
    humid: float
    light: float
    co2: float

class FreshnessInput(BaseModel):
    food: str
    temp: float
    humid: float
    prepared_time: str  # ISO format

class SurplusInput(BaseModel):
    food: str
    people: int

# =========================
# SPOILAGE PREDICTION API
# =========================
@app.post("/predict")
def predict_spoilage(data: SpoilageInput):

    # ✅ REALISTIC VALIDATION
    if data.temp < -30 or data.temp > 60:
        raise HTTPException(
            status_code=400,
            detail="Temperature out of realistic range (-30 to 60°C)"
        )

    if data.humid < 0 or data.humid > 100:
        raise HTTPException(
            status_code=400,
            detail="Humidity must be between 0 and 100"
        )

    humidity_risk = int(data.humid > 90)

    df = pd.DataFrame([{
        "Temp": data.temp,
        "Humid": data.humid,
        "Light": data.light,
        "CO2": data.co2,
        "humidity_risk": humidity_risk
    }])

    prediction = model.predict(df)[0]

    return {
        "result": "Good (Safe)" if prediction == 0 else "Bad (Spoiled)"
    }

# =========================
# FRESHNESS DURATION API
# =========================
@app.post("/freshness")
def predict_freshness(data: FreshnessInput):

    try:
        prepared_time = datetime.fromisoformat(data.prepared_time)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="prepared_time must be in ISO format (YYYY-MM-DDTHH:MM:SS)"
        )

    remaining, status = freshness_duration(
        data.food,
        data.temp,
        data.humid,
        prepared_time
    )

    return {
        "food": data.food,
        "remaining_hours": remaining,
        "status": status
    }

# =========================
# SURPLUS ESTIMATION API
# =========================
@app.post("/estimate")
def estimate_food(data: SurplusInput):
    food = data.food.lower()

    if food not in BASE_SERVING:
        return {
            "error": "Food not supported yet",
            "suggestion": "Use common foods like rice, biryani, dosa, chapathi, idli"
        }

    base = BASE_SERVING[food]
    quantity = base * data.people
    unit = "kg" if base < 1 else "pieces"

    return {
        "food": food,
        "people": data.people,
        "recommended_quantity": round(quantity, 2),
        "unit": unit
    }