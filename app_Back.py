from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

app = FastAPI(title="Food Wastage Reduction & Estimation API")

# =========================
# BASE FRESHNESS HOURS
# =========================
BASE_FRESHNESS = {
    "rice": 6,
    "chapathi": 6,
    "chapati": 6,
    "biryani": 8,
    "curry": 8,
    "fruit": 12,
    "packed": 24
}

# =========================
# BASE SERVING TABLE (per person)
# =========================
BASE_SERVING = {
    # Veg
    "rice": 0.25,
    "biryani": 0.35,
    "chapathi": 2.5,
    "chapati": 2.5,
    "roti": 2.5,
    "dosa": 2,
    "idli": 3.5,
    "vada": 3,
    "curry": 0.20,
    "paneer": 0.15,
    "chana": 0.18,
    "rajma": 0.18,
    "dal": 0.15,
    "aloo": 0.20,
    "gobi": 0.18,
    "bhindi": 0.15,
    "palak": 0.20,
    "mushroom": 0.20,
    # Non-veg
    "chicken": 0.20,
    "chicken curry": 0.22,
    "butter chicken": 0.20,
    "mutton": 0.18,
    "mutton curry": 0.20,
    "egg": 3,
    "egg curry": 0.20,
    "fish": 0.20,
    "fish curry": 0.22,
    "keema": 0.18
}

# =========================
# EXACT INGREDIENTS (per person) - with quantified spices
# =========================
INGREDIENTS_PER_PERSON = {
    "rice": {
        "raw_rice": 0.100,
        "water": "0.2–0.25 L"
    },
    "biryani": {
        "raw_rice": 0.120,
        "meat_or_veg": 0.150,
        "onion": 0.080,
        "tomato": 0.060,
        "yogurt": 0.050,
        "oil_ghee": 0.030,
        "turmeric_powder": 0.002,
        "red_chili_powder": 0.003,
        "coriander_powder": 0.004,
        "garam_masala": 0.003,
        "biryani_masala": 0.005
    },
    "chapathi": {
        "whole_wheat_flour": 0.060,
        "water": "as needed"
    },
    "dosa": {
        "rice": 0.060,
        "urad_dal": 0.020,
        "oil": 0.010
    },
    "curry": {
        "main_ingredient": 0.150,
        "onion": 0.050,
        "tomato": 0.050,
        "oil": 0.020,
        "ginger_garlic_paste": 0.015,
        "turmeric_powder": 0.0015,
        "red_chili_powder": 0.0025,
        "coriander_powder": 0.004,
        "garam_masala": 0.002
    },
    "paneer": {
        "paneer": 0.150,
        "onion": 0.060,
        "tomato": 0.070,
        "capsicum": 0.040,
        "cream": 0.030,
        "oil_ghee": 0.020,
        "turmeric_powder": 0.0015,
        "red_chili_powder": 0.003,
        "coriander_powder": 0.004,
        "garam_masala": 0.003
    },
    "chicken curry": {
        "chicken": 0.150,
        "onion": 0.060,
        "tomato": 0.070,
        "yogurt": 0.040,
        "oil": 0.025,
        "ginger_garlic_paste": 0.015,
        "turmeric_powder": 0.002,
        "red_chili_powder": 0.004,
        "coriander_powder": 0.005,
        "garam_masala": 0.003
    },
    "butter chicken": {
        "chicken": 0.150,
        "tomato": 0.100,
        "cream": 0.040,
        "butter": 0.020,
        "cashew_paste": 0.015,
        "turmeric_powder": 0.001,
        "red_chili_powder": 0.003,
        "garam_masala": 0.003,
        "kasuri_methi": 0.002
    },
    "mutton curry": {
        "mutton": 0.150,
        "onion": 0.070,
        "tomato": 0.060,
        "yogurt": 0.050,
        "oil_ghee": 0.030,
        "ginger_garlic_paste": 0.020,
        "turmeric_powder": 0.002,
        "red_chili_powder": 0.004,
        "coriander_powder": 0.005,
        "garam_masala": 0.004
    },
    "egg curry": {
        "eggs": 3,
        "onion": 0.050,
        "tomato": 0.060,
        "oil": 0.020,
        "turmeric_powder": 0.0015,
        "red_chili_powder": 0.003,
        "coriander_powder": 0.004,
        "garam_masala": 0.002
    },
    "fish curry": {
        "fish": 0.150,
        "onion": 0.050,
        "tomato": 0.060,
        "coconut_milk": 0.050,
        "oil": 0.020,
        "turmeric_powder": 0.002,
        "red_chili_powder": 0.004,
        "coriander_powder": 0.004,
        "tamarind_paste": 0.010
    },
    "keema": {
        "minced_meat": 0.150,
        "onion": 0.060,
        "tomato": 0.060,
        "peas": 0.030,
        "oil": 0.020,
        "turmeric_powder": 0.0015,
        "red_chili_powder": 0.003,
        "coriander_powder": 0.004,
        "garam_masala": 0.003
    }
}

# Common spices used in guessing function
COMMON_SPICES = {
    "turmeric_powder": 0.0015,
    "red_chili_powder": 0.003,
    "coriander_powder": 0.004,
    "garam_masala": 0.0025
}

# =========================
# NUTRITION PER 100g prepared/cooked dish (approximate averages)
# =========================
NUTRITION_PER_100G = {
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28.0, "fat": 0.3, "fiber": 0.4},
    "biryani": {"calories": 180, "protein": 8.0, "carbs": 25.0, "fat": 6.0, "fiber": 1.5},
    "chapathi": {"calories": 297, "protein": 9.0, "carbs": 46.0, "fat": 7.5, "fiber": 4.0},
    "chapati": {"calories": 297, "protein": 9.0, "carbs": 46.0, "fat": 7.5, "fiber": 4.0},
    "roti": {"calories": 297, "protein": 9.0, "carbs": 46.0, "fat": 7.5, "fiber": 4.0},
    "dosa": {"calories": 168, "protein": 4.0, "carbs": 30.0, "fat": 3.5, "fiber": 1.5},
    "idli": {"calories": 150, "protein": 5.0, "carbs": 30.0, "fat": 1.0, "fiber": 2.0},
    "curry": {"calories": 140, "protein": 6.0, "carbs": 12.0, "fat": 8.0, "fiber": 3.0},
    "paneer": {"calories": 265, "protein": 18.0, "carbs": 3.0, "fat": 20.0, "fiber": 0.5},
    "chicken curry": {"calories": 180, "protein": 20.0, "carbs": 8.0, "fat": 9.0, "fiber": 2.0},
    "butter chicken": {"calories": 220, "protein": 18.0, "carbs": 10.0, "fat": 14.0, "fiber": 1.5},
    "mutton curry": {"calories": 210, "protein": 20.0, "carbs": 6.0, "fat": 13.0, "fiber": 1.5},
    "egg curry": {"calories": 170, "protein": 12.0, "carbs": 6.0, "fat": 11.0, "fiber": 1.0},
    "fish curry": {"calories": 170, "protein": 20.0, "carbs": 5.0, "fat": 8.0, "fiber": 1.0},
    "keema": {"calories": 200, "protein": 22.0, "carbs": 8.0, "fat": 10.0, "fiber": 2.0},
}

# =========================
# GUESSING LOGIC - with quantified spices
# =========================
def guess_ingredients(food: str) -> dict | None:
    f = food.lower().strip()
    if "mushroom" in f:
        return {
            "mushrooms": 0.200,
            "onion": 0.060,
            "tomato": 0.080,
            "ginger_garlic_paste": 0.015,
            "oil": 0.020,
            **COMMON_SPICES,
            "kasuri_methi": 0.002
        }
    if "paneer" in f:
        return {
            "paneer": 0.150,
            "onion": 0.060,
            "tomato": 0.070,
            "cream": 0.030,
            "oil_ghee": 0.020,
            **COMMON_SPICES
        }
    if any(x in f for x in ["chicken", "murgh"]):
        if "butter" in f or "makhani" in f:
            return {
                "chicken": 0.150,
                "tomato": 0.100,
                "cream": 0.040,
                "butter": 0.020,
                "cashew_paste": 0.015,
                **COMMON_SPICES,
                "kasuri_methi": 0.002
            }
        else:
            return {
                "chicken": 0.150,
                "onion": 0.060,
                "tomato": 0.070,
                "yogurt": 0.040,
                "oil": 0.025,
                "ginger_garlic_paste": 0.015,
                **COMMON_SPICES
            }
    if any(x in f for x in ["mutton", "lamb", "goat"]):
        return {
            "mutton": 0.150,
            "onion": 0.070,
            "tomato": 0.060,
            "yogurt": 0.050,
            "oil_ghee": 0.030,
            "ginger_garlic_paste": 0.020,
            **COMMON_SPICES
        }
    if any(x in f for x in ["egg", "anda"]):
        return {
            "eggs": 3,
            "onion": 0.050,
            "tomato": 0.060,
            "oil": 0.020,
            **COMMON_SPICES
        }
    if any(x in f for x in ["fish", "machli"]):
        return {
            "fish": 0.150,
            "onion": 0.050,
            "tomato": 0.060,
            "oil": 0.020,
            **COMMON_SPICES,
            "coconut_milk": 0.050
        }
    if any(word in f for word in ["curry", "masala", "sabzi", "nonveg", "meat", "korma"]):
        return {
            "main_ingredient": 0.150,
            "onion": 0.060,
            "tomato": 0.060,
            "oil": 0.025,
            "ginger_garlic_paste": 0.015,
            **COMMON_SPICES
        }
    return None

# =========================
# FRESHNESS CALCULATION
# =========================
def freshness_duration(food: str, temp: float, humid: float, prepared_time: datetime):
    food = food.lower().strip()
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
# INPUT MODELS
# =========================
class FreshnessInput(BaseModel):
    food: str
    temp: float
    humid: float
    prepared_time: str

class SurplusInput(BaseModel):
    food: str
    people: int

# =========================
# ENDPOINTS
# =========================

@app.post("/freshness")
def predict_freshness(data: FreshnessInput):
    try:
        prep_time = datetime.fromisoformat(data.prepared_time)
    except ValueError:
        raise HTTPException(400, "prepared_time must be ISO format (YYYY-MM-DDTHH:MM:SS)")

    remaining, status = freshness_duration(data.food, data.temp, data.humid, prep_time)
    return {"food": data.food, "remaining_hours": remaining, "status": status}

@app.post("/estimate")
def estimate_food(data: SurplusInput):
    food = data.food.lower().strip()
    people = data.people

    note = (
        "Spice quantities are approximate — adjust according to taste preference (mild/medium/spicy). "
        "Nutritional values are approximate averages for the prepared dish — actual values vary by recipe, oil/ghee amount, etc."
    )

    if food in BASE_SERVING:
        base_qty = BASE_SERVING[food]
        unit = "kg" if base_qty < 1 else "pieces"
        ingredients = INGREDIENTS_PER_PERSON.get(food, INGREDIENTS_PER_PERSON.get("curry", {}))
        nutrition_key = food
    else:
        guessed = guess_ingredients(food)
        if guessed:
            base_qty = 0.20
            unit = "kg"
            ingredients = guessed
            note += " This is an estimated list based on common Indian-style recipes."
            nutrition_key = None
        else:
            return {
                "error": "Food not recognized",
                "supported_exact": sorted(list(BASE_SERVING.keys())),
                "examples": [
                    "butter chicken", "chicken tikka masala", "mutton curry",
                    "egg masala", "fish curry", "keema matar", "paneer butter masala",
                    "mushroom masala", "aloo gobi", "dal tadka"
                ]
            }

    total_qty_kg = base_qty * people

    # Nutrition calculation
    nutrition_estimate = {"message": "Nutrition data not available for this food yet"}
    
    if nutrition_key and nutrition_key in NUTRITION_PER_100G:
        per_100g = NUTRITION_PER_100G[nutrition_key]
        multiplier = total_qty_kg * 10
        
        nutrition_estimate = {
            "total_calories_kcal": round(per_100g["calories"] * multiplier),
            "total_protein_g": round(per_100g["protein"] * multiplier, 1),
            "total_carbohydrates_g": round(per_100g["carbs"] * multiplier, 1),
            "total_fat_g": round(per_100g["fat"] * multiplier, 1),
            "total_fiber_g": round(per_100g["fiber"] * multiplier, 1),
            "per_100g_reference": {
                "calories": per_100g["calories"],
                "protein": per_100g["protein"],
                "carbohydrates": per_100g["carbs"],
                "fat": per_100g["fat"],
                "fiber": per_100g["fiber"]
            },
            "note": "Calculated for the full recommended quantity. Does not include side dishes (rice, roti, etc.)."
        }

    response = {
        "food": food,
        "people_count": people,
        "recommended_total_quantity": round(total_qty_kg, 2),
        "unit": unit,
        "ingredients_per_person": {},
        "total_ingredients_estimate": {},
        "nutrition_estimate": nutrition_estimate,
        "note": note
    }

    for ing, val in ingredients.items():
        if isinstance(val, (int, float)):
            total_val = val * people
            response["ingredients_per_person"][ing] = round(val, 4)
            unit_str = "kg" if val >= 0.01 else "approx kg (small quantities)"
            response["total_ingredients_estimate"][ing] = {
                "quantity": round(total_val, 3),
                "unit": unit_str
            }
        else:
            response["ingredients_per_person"][ing] = val
            response["total_ingredients_estimate"][ing] = val

    return response

@app.get("/")
def root():
    return {"message": "Food Wastage Reduction & Estimation API is running"}