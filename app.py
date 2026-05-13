from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Drug Regulatory Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load artifacts at startup from the app directory
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "drug_classifier_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
imputer = joblib.load(BASE_DIR / "imputer.pkl")
le = joblib.load(BASE_DIR / "label_encoder.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

class DrugInput(BaseModel):
    Dosage_mg: float
    Price_Per_Unit: float
    Production_Cost: float
    Marketing_Spend: float
    Clinical_Trial_Phase: int
    Side_Effect_Severity_Score: float
    Abuse_Potential_Score: float
    Prescription_Rate: float
    Hospital_Distribution_Percentage: float
    Pharmacy_Distribution_Percentage: float
    Annual_Sales_Volume: float
    Regulatory_Risk_Score: float
    Approval_Time_Months: int
    Patent_Duration_Years: int
    RD_Investment_Million: float
    Competitor_Count: int
    Recall_History_Count: int
    Adverse_Event_Reports: int
    Drug_Form: str
    Therapeutic_Class: str
    Manufacturing_Region: str
    Requires_Cold_Storage: str
    OTC_Flag: str
    High_Risk_Substance: str
    Insurance_Coverage_Percentage: float
    Export_Percentage: float
    Online_Sales_Percentage: float
    Brand_Reputation_Score: float
    Doctor_Recommendation_Rate: float

def preprocess(input_data: DrugInput) -> pd.DataFrame:
    data = input_data.dict()
    # Rename to match training feature name
    data['R&D_Investment_Million'] = data.pop('RD_Investment_Million')

    # Binary encode
    for col in ['Requires_Cold_Storage', 'OTC_Flag', 'High_Risk_Substance']:
        data[col] = 1 if data[col] == 'Yes' else 0

    df = pd.DataFrame([data])

    # One-hot encode
    df = pd.get_dummies(df)

    # Align columns with training set
    df = df.reindex(columns=feature_columns, fill_value=0)

    df_imputed = pd.DataFrame(imputer.transform(df), columns=feature_columns)
    df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=feature_columns)
    return df_scaled

@app.post("/predict")
def predict(drug: DrugInput):
    processed = preprocess(drug)
    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]
    label = le.inverse_transform([pred])[0]
    return {
        "prediction": label,
        "regulated_probability": round(float(prob), 4),
        "confidence": "High" if max(prob, 1-prob) > 0.85 else "Medium"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)