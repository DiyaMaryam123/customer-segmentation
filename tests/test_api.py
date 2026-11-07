from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"Income": 60, "SpendingScore": 50}
    )
    assert response.status_code == 200
    assert "assigned_cluster" in response.json()


def test_invalid_income():
    response = client.post(
        "/predict",
        json={"Income": -5, "SpendingScore": 50}
    )
    assert response.status_code == 422


def test_invalid_spending_score():
    response = client.post(
        "/predict",
        json={"Income": 60, "SpendingScore": 150}
    )
    assert response.status_code == 422
