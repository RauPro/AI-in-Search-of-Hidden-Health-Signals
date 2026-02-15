import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
# Initialize the App
app = FastAPI(title="Health Hackathon Prediction API")

# Define the Path to Saved Models
MODEL_DIR = Path("saved_models")

# Global variable to store loaded models (so we load them only once)
models = {}

# ---------------------------------------------------------
# 1. LOAD MODELS ON STARTUP
# ---------------------------------------------------------
@app.on_event("startup")
def load_models():
    """
    Load all disease expert ensembles into memory when the server starts.
    """
    disease_list = [
        "diabetes", "cvd", "stroke", "lung", "cancer", 
        "hibp", "arthritis", "memory", "psychiatric" # Add others if you trained them
    ]
    
    print("--- Loading Expert Ensembles ---")
    for disease in disease_list:
        path = MODEL_DIR / f"ensemble_{disease}.pkl"
        if path.exists():
            models[disease] = joblib.load(path)
            print(f" [OK] Loaded {disease.upper()} Expert")
        else:
            print(f" [WARNING] Could not find model for {disease} at {path}")

# ---------------------------------------------------------
# 2. DEFINE INPUT DATA FORMAT (Pydantic)
# ---------------------------------------------------------
from typing import Optional
from pydantic import BaseModel

class PatientData(BaseModel):
    # --- 1. Core Identifiers ---
    age: float
    female: int              # 0=Male, 1=Female
    ethnicity: str = "White" # Default to avoid LightGBM crashes
    
    # --- 2. Physical Vitals ---
    # We allow weight/height so we can calc BMI automatically if BMI is missing
    height: Optional[float] = None  # in meters
    weight: Optional[float] = None  # in kg
    bmi: Optional[float] = None
    
    # --- 3. The "Time Machine" (History) ---
    # Critical for the "Velocity" features that won you the Gold Medal
    bmi_lag1: Optional[float] = None        # Weight/BMI 2 years ago
    self_rated_health_lag1: Optional[float] = None # Health 2 years ago
    
    # --- 4. The Full Screening List (From your Dictionary) ---
    # We default these to 0 or None so the API is flexible
    
    self_rated_health: float = 3.0  # 1=Excellent, 5=Poor (Default Average)
    
    mobility: float = 0.0      # Difficulty walking? (0=No, >0=Yes)
    gross_motor: float = 0.0   # Large muscle movement
    fine_motor: float = 0.0    # Small muscle movement
    large_muscle: float = 0.0
    
    adl: float = 0.0           # Activities of Daily Living score
    iadl: float = 0.0          # Instrumental ADL score
    
    cognition: float = 20.0    # Cog Score (0-35). Default to Healthy (20+)
    memory_recall: float = 5.0 # Memory test score
    serial7: float = 5.0       # Serial 7s test score
    
    cesd: float = 0.0          # Depression Score (0=None, 8=Severe)
    depressed: int = 0         # 0/1 flag
    lonely: int = 0            # 0/1 flag
    restless_sleep: int = 0    # 0/1 flag
    effort: int = 0            # "Everything was an effort" flag
    
    ever_smoked: int = 0       # 0=No, 1=Yes
    current_smoker: int = 0    # 0=No, 1=Yes
    drinks_per_day: float = 0.0
    drink_days_week: float = 0.0
    vigorous_activity: int = 0 # Frequency (1-5)
    
    working: int = 0           # 0=No, 1=Yes

# ---------------------------------------------------------
# 3. PREDICTION LOGIC
# ---------------------------------------------------------
def predict_disease_risk(disease, data_dict):
    """
    Run the specific ensemble (Cat+LGB+XGB) for one disease.
    """
    if disease not in models:
        return None
    
    ensemble = models[disease]
    
    # 1. Create DataFrame from input
    df = pd.DataFrame([data_dict])
    
    # =========================================================
    # PART 1: REPLICATE "EXTRACT_SCREENING_FEATURES" LOGIC
    # =========================================================
    # We must create the variables the model learned from your function
    
    # A. Age Squared (Non-linear aging effect)
    if "age" in df.columns:
        df["age_squared"] = df["age"] ** 2
        
    # B. Screening Date (Simulate "Today")
    current_date = datetime.now()
    df["screening_year"] = float(current_date.year)
    df["screening_month"] = float(current_date.month)
    
    # C. Time Gaps (Assume standard 2-year gap if unknown)
    # The model expects "years_since_last_screening". 
    # Since this is a new prediction, we assume a standard interval.
    df["years_since_last_screening"] = 2.0 
    
    # D. Education (Default to High School/12 years if missing)
    # Your model uses 'education', 'edu_cat', 'degree'
    if "education" not in df.columns:
        df["education"] = 12.0  # Default: 12 years
        df["edu_cat"] = 1.0     # Default: High School GED
        df["degree"] = 1.0      # Default: High School Diploma
        
    # =========================================================
    # PART 2: REPLICATE "TEMPORAL_FEATURES" LOGIC
    # =========================================================
    
    # A. BMI Velocity
    if 'bmi_lag1' in df.columns and df['bmi_lag1'][0] is not None:
        df['bmi_velocity_2yr'] = (df['bmi'] - df['bmi_lag1']) / 2.0
    else:
        df['bmi_velocity_2yr'] = 0.0

    # =========================================================
    # PART 3: THE CATEGORICAL METADATA FIX (Essential)
    # =========================================================
    
    # A. Define the EXACT categories from your extraction script
    expected_ethnicities = ["White", "Black", "Hispanic", "Other"]
    
    # B. Map Numbers to Strings (if frontend sends 1,2,3)
    ethnicity_map = {1: "White", 2: "Black", 3: "Hispanic", 4: "Other"}
    if "ethnicity" in df.columns:
        val = df["ethnicity"].iloc[0]
        if isinstance(val, int) or (isinstance(val, str) and val.isdigit()):
            df["ethnicity"] = ethnicity_map.get(int(val), "Other")
            
    # C. Force the Category Type
    if "ethnicity" in df.columns:
        df["ethnicity"] = pd.Categorical(
            df["ethnicity"], 
            categories=expected_ethnicities
        )
    else:
        # Create default if missing
        df["ethnicity"] = pd.Categorical(
            ["White"], 
            categories=expected_ethnicities
        )

    # =========================================================
    # PART 4: FILL MISSING COLUMNS & PREDICT
    # =========================================================
    
    # Get required feature names from LightGBM
    try:
        required_cols = ensemble['lgb'].booster_.feature_name()
    except:
        required_cols = ensemble['lgb'].feature_name_
        
    # Fill any remaining missing columns with 0.0
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 
            
    # Select columns in correct order
    X_input = df[required_cols]
    
    # Get probabilities
    p_lgb = ensemble['lgb'].predict_proba(X_input)[:, 1][0]
    p_cat = ensemble['cat'].predict_proba(X_input)[:, 1][0]
    p_xgb = ensemble['xgb'].predict_proba(X_input)[:, 1][0]
    
    # Average them
    avg_risk = (p_cat + p_lgb + p_xgb) / 3.0
    return float(avg_risk)

# ---------------------------------------------------------
# 4. THE ENDPOINT
# ---------------------------------------------------------
@app.post("/predict")
def predict_health_risks(patient: PatientData):
    """
    Frontend sends patient data -> We return risk % for all diseases.
    """
    data = patient.dict()
    results = {}
    
    # 1. Predict for every loaded disease
    for disease in models.keys():
        risk_score = predict_disease_risk(disease, data)
        if risk_score is not None:
            results[disease] = round(risk_score * 100, 2) # Return as percentage
            
    # 2. Calculate "Overall Health Score" (Inverse of average risk)
    # Simple logic for the UI: 100 - Max_Risk
    if results:
        max_risk = max(results.values())
        health_score = max(0, 100 - max_risk)
    else:
        health_score = 0
        
    return {
        "status": "success",
        "health_score": round(health_score, 1),
        "risks": results,
        "message": "Prediction successful. Remember this is a screening tool, not a diagnosis."
    }
