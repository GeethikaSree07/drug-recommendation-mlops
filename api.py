from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import joblib
import pandas as pd

app = FastAPI(title="Drug Recommendation API")

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "Data"
MONITORING_DIR = BASE_DIR / "monitoring"
LOG_FILE = MONITORING_DIR / "prediction_logs.csv"
REFERENCE_FILE = DATA_DIR / "drug200.csv"
REPORT_FILE = MONITORING_DIR / "live_drift_report.html"

MONITORING_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)

model = joblib.load(
    BASE_DIR / "models" / "drug_recommendation_model.pkl"
)


class PatientData(BaseModel):
    Age: int
    Sex: str
    BP: str
    Cholesterol: str
    Na_to_K: float


FEATURES = [
    "Age",
    "Sex",
    "BP",
    "Cholesterol",
    "Na_to_K"
]


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


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

    log_row = input_data.copy()
    log_row["timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    write_header = (
        not LOG_FILE.exists()
        or LOG_FILE.stat().st_size == 0
    )

    log_row.to_csv(
        LOG_FILE,
        mode="a",
        header=write_header,
        index=False
    )

    return {
        "predicted_drug_class": str(prediction),
        "notice": (
            "Educational ML prototype only; not medical advice."
        )
    }


@app.get("/monitoring/report")
def monitoring_report():
    if not REFERENCE_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Reference dataset not found.",
                "reference_path": str(REFERENCE_FILE),
                "data_dir_exists": DATA_DIR.exists(),
                "base_dir": str(BASE_DIR),
                "root_files": [
                    p.name for p in BASE_DIR.iterdir()
                ],
                "data_dir_files": (
                    [p.name for p in DATA_DIR.iterdir()]
                    if DATA_DIR.exists()
                    else []
                )
            }
        )

    if not LOG_FILE.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "No prediction inputs logged yet. "
                "Make a test prediction first."
            )
        )

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        reference = pd.read_csv(REFERENCE_FILE)[FEATURES]
        current = pd.read_csv(LOG_FILE)[FEATURES]

        if len(current) < 5:
            raise HTTPException(
                status_code=400,
                detail=(
                    "At least 5 logged predictions are "
                    "needed for this demo report."
                )
            )

        report = Report([DataDriftPreset()])

        result = report.run(
            reference_data=reference,
            current_data=current
        )

        result.save_html(str(REPORT_FILE))

        return FileResponse(
            REPORT_FILE,
            media_type="text/html",
            filename="live_drift_report.html"
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not generate drift report: {exc}"
            )
        )
