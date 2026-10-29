from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Drug Recommendation API")

app.mount("/static", StaticFiles(directory="frontend"), name="static")

model = joblib.load("models/drug_recommendation_model.pkl")


class PatientData(BaseModel):
    Age: int
    Sex: str
    BP: str
    Cholesterol: str
    Na_to_K: float


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


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