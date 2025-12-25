import pytest
from fastapi.testclient import TestClient

from src.main import app


# FIXTURE: To jest "włącznik" aplikacji dla testów.
# Dzięki "with TestClient(app) as c", odpalamy zdarzenia "lifespan" (ładowanie modelu).
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Sprawdza czy endpoint /health zwraca 200 OK i status"""
    response = client.get("/health")
    assert response.status_code == 200
    # Teraz running_model musi być True, bo "with TestClient" go załadował
    assert response.json() == {"status": "ok", "running_model": True}


def test_recommendation_flow(client):
    """Sprawdza czy endpoint /recommend zwraca listę napisów"""
    user_id = 10
    response = client.get(f"/recommend/{user_id}")

    # Assertions
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == user_id
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
