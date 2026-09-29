from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Drug Recommendation API")

model = joblib.load("models/drug_recommendation_model.pkl")


class PatientData(BaseModel):
    Age: int
    Sex: str
    BP: str
    Cholesterol: str
    Na_to_K: float


@app.get("/")
def home():
    return {"message": "Drug Recommendation API is running"}


@app.post("/predict")
def predict(data: PatientData):
    input_data = pd.DataFrame([{
        "Age": data.Age,
        "Sex": data.Sex,
        "BP": data.BP,
        "Cholesterol": data.Cholesterol,
        "Na_to_K": data.Na_to_K
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_drug_class": prediction
    }