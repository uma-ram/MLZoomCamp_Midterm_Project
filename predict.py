import pickle
import pandas as pd
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.feature_extraction import DictVectorizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal 
from custom_transformer import DictVectorizerTransformer


app = FastAPI(title="Medical Insurance Cost Prediction API")

# ---------------------------------------------------------
# Class for request - implemented validation using pydantic
# ---------------------------------------------------------
class MedicalCostInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the person (18–100)")
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=15.0, le=60.0, description="Body Mass Index")
    children: int = Field(..., ge=0, le=5, description="Number of children/dependents")    
    smoker: Literal["yes", "no"]
    region: Literal["southwest", "southeast", "northwest", "northeast"]


# ---------------------------------------------------------
# Load Trained model - Returns model
# ---------------------------------------------------------

def load_model(model_path="model.pkl"):
    print("Loading model...")
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)
    return model

@app.get("/")
def root():
    return {"message": "Medical Insurance Cost Prediction API is running!"}

# ---------------------------------------------------------
# API - Predict Function - Returns predicted medical cost
# ---------------------------------------------------------

@app.post("/predict")
def predict(data: MedicalCostInput):

    try:
        print("Loading...")

        model = load_model()

        print("Predicting...")
        # models pipeline handles preprocessing

        df = pd.DataFrame([data.model_dump()])  # or data.dict()

        # Convert categorical columns to string explicitly (important!)
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        prediction = model.predict(df)[0] 

        return {
            "predicted_medical_cost": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")



