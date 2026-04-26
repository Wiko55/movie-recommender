import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from src.data_loader import load_data
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
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "redis_connected": True,
    }


def test_recommendation_flow(client):
    """Sprawdza czy endpoint /recommend zwraca listę napisów"""
    with TestClient(app) as client:
        user_id = 1
        response = client.get(f"/recommend/{user_id}")

        if response.status_code != 200:
            print(f"\nBŁĄD API: {response.json()}")

        assert response.status_code == 200

        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0


def test_user_id_collision(client):
    """Sprawdza czy ID w bazie SQL nie koliduje z id w CSV"""
    _, ratings, _ = load_data()
    unique_user = f"TestUser_{int(time.time())}"
    response = client.post(
        "/register", json={"username": unique_user, "password": "testpassword123"}
    )
    assert response.status_code == 200
    assert "user_id" in response.json()
    data = response.json()
    user_id = data["user_id"]
    print(f"\n[TEST INFO] Zarejestrowano User ID: {user_id}")
    assert user_id > max(ratings["userId"]), (
        f"Nowe ID użytkownika {user_id} koliduje z istniejącymi ID w CSV"
    )


def test_retraining_endpoint(client):
    """Sprawdza czy endpoint /retrain działa poprawnie"""
    response = client.post("/admin/retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Ponowne trenowanie modelu zostało uruchomione w tle."
