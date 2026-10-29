from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Drug Recommendation System" in response.text


def test_prediction():
    data = {
        "Age": 30,
        "Sex": "F",
        "BP": "HIGH",
        "Cholesterol": "NORMAL",
        "Na_to_K": 15.5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "predicted_drug_class" in response.json()