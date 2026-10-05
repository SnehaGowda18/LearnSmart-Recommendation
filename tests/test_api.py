from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["users"] == 100
    assert data["courses"] == 20
    assert data["model"] == "hybrid"
    assert data["content_weight"] == 0.3
    assert data["collaborative_weight"] == 0.7


def test_recommend():
    response = client.post(
        "/recommend",
        json={
            "user_id": "U001",
            "top_n": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["user_id"] == "U001"
    assert data["top_n"] == 5
    assert data["model"] == "hybrid"
    assert len(data["recommendations"]) == 5


def test_invalid_user():
    response = client.post(
        "/recommend",
        json={
            "user_id": "INVALID",
            "top_n": 5,
        },
    )

    assert response.status_code == 404