from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.dataprocessing import load_and_process
from src.recommender import MovieRecommender

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pobieranie danych z serwera")
    matrix = load_and_process()
    rec = MovieRecommender(n_clusters=5)
    rec.fit(matrix=matrix)
    models["recommender"] = rec

    print("Serwer gotowy")
    yield
    models.clear()


app = FastAPI(title="Recommender API", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "strona zyje", "messege": "API dziala poprawnie"}


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int) -> dict:
    rec = models["recommender"]
    if not rec:
        raise HTTPException(status_code=500, detail="Model niedostępny")
    recommendations = rec.recommend(user_id)
    if not recommendations:
        raise HTTPException(
            status_code=404, detail="Nie ma takiego uzytkownika w bazie"
        )
    return {"user_id": user_id, "recommendations": recommendations}
