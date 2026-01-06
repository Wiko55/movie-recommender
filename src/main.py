import json
import logging
import os
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_data
from src.recommender import MovieRecommender
from src.tmdb_client import fetch_posters_for_movies

# ------------------------------

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguracja Redisa
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Globalny słownik na modele
ml_models = {}
redis_client = None
CACHE_MAX_ITEMS = 50
movies_data = None
links_data = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. START APLIKACJI
    logger.info("🚀 Uruchamianie Systemu Rekomendacji...")
    global redis_client, movies_data, links_data
    # --- ŁADOWANIE NOWEGO MODELU ---
    try:
        logger.info("📥 Pobieranie i ładowanie danych MovieLens...")
        movies, ratings, links = load_data()
        movies_data = movies
        links_data = links
        logger.info("🧠 Trenowanie modelu k-NN...")
        recommender = MovieRecommender()
        recommender.fit(movies, ratings)

        ml_models["latest"] = recommender
        logger.info("✅ Model gotowy do pracy!")
    except Exception as e:
        logger.error(f"❌ Błąd krytyczny podczas ładowania modelu: {e}")
        ml_models["latest"] = None
    # -------------------------------

    # Redis Connection
    global redis_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        if redis_client.ping():
            logger.info(f"✅ Połączono z Redisem na {REDIS_HOST}")
        else:
            logger.warning("⚠️ Redis nie odpowiada (ale połączenie nawiązane).")
    except Exception as e:
        logger.warning(f"⚠️ Nie można połączyć z Redisem: {e}. Działamy bez Cache.")
        redis_client = None

    yield  # Tutaj aplikacja działa i obsługuje zapytania...

    # 2. ZAMYKANIE APLIKACJI
    ml_models.clear()
    if redis_client:
        redis_client.close()
    logger.info("🛑 Aplikacja zatrzymana.")


app = FastAPI(lifespan=lifespan, title="Movie Recommender API")


# Modele Pydantic
class MovieItem(BaseModel):
    title: str
    poster: str | None = None


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[MovieItem]
    source: str
    model_version: str


@app.get("/", summary="Read Root")
def read_root():
    return {"message": "Witaj w API Rekomendacji Filmów (Wersja ML)"}


@app.get("/health", response_model=HealthCheck, summary="Health Check")
def health_check():
    return HealthCheck(
        status="ok",
        model_loaded="latest" in ml_models and ml_models["latest"] is not None,
        redis_connected=redis_client is not None and redis_client.ping(),
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_n: int = 5):
    """
    Zwraca rekomendacje na podstawie Collaborative Filtering.
    """
    limit = min(top_n, CACHE_MAX_ITEMS)
    cache_key = f"rec_img_v1_{user_id}"  # Zmieniamy klucz na v2, bo mamy nowy model!

    # 1. CACHE (Redis)
    try:
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return RecommendationResponse(**data)
    except Exception as e:
        logger.error(f"Redis error: {e}")

    # 2. MODEL ML (Obliczenia)
    model = ml_models.get("latest")
    if not model:
        raise HTTPException(status_code=503, detail="Model ML nie jest gotowy")

    # Wywołanie nowego recommendera
    # UWAGA: Nasz nowy model zwraca listę tytułów (stringów), co pasuje do modelu odpowiedzi!
    try:
        titles = model.recommend(user_id, top_n=CACHE_MAX_ITEMS)
    except Exception as e:
        logger.error(f"Błąd modelu: {e}")

    # 3. Posters
    if not titles:
        logger.info(f"Cold Start dla User {user_id} - serwuję Hity Globalne")
        titles = model.get_popular_movies(top_n=CACHE_MAX_ITEMS)
        logger.info(f" {user_id} - serwuję Hity Globalne")
        source = "popularity_fallback"

    if titles:
        final_items = fetch_posters_for_movies(
            movies_data[movies_data["title"].isin(titles)], links_data, top_n=limit
        )
    # 4. ZAPIS DO CACHE
    if redis_client and final_items:
        payload = {
            "user_id": user_id,
            "recommendations": final_items,
            "source": "model_computation",
            "model_version": "v2_knn",
        }
        try:
            redis_client.setex(
                cache_key, 60 * 10, json.dumps(payload)
            )  # Cache na 10 min
        except Exception:
            pass

    return RecommendationResponse(
        user_id=user_id,
        recommendations=final_items,
        source="model_computation",
        model_version="v2_knn",
    )


@app.get("/similar/{movie_title}", response_model=RecommendationResponse)
async def get_similar_movies(movie_title: str, top_n: int = 5):
    """
    Rekomenduje filmy podobne do podanego tytułu.
    """
    limit = min(top_n, CACHE_MAX_ITEMS)
    # Prosty cache key
    cache_key = f"rec_movie_{movie_title.replace(' ', '_').lower()}"

    # 1. CACHE
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return RecommendationResponse(**json.loads(cached))

    # 2. MODEL
    model = ml_models.get("latest")
    if not model or movies_data is None:
        raise HTTPException(status_code=503, detail="System niegotowy")

    titles = model.get_recommendations_for_movie(movie_title, top_n=limit)

    # 3. PLAKATY
    if not titles:
        final_items = []
    else:
        final_items = fetch_posters_for_movies(
            movies_data[movies_data["title"].isin(titles)], links_data, top_n=limit
        )

    response = {
        "user_id": 0,
        "recommendations": final_items,
        "source": "item_item_similarity",
        "model_version": "v2_knn",
    }

    # 4. ZAPIS DO CACHE
    if redis_client:
        redis_client.setex(cache_key, 600, json.dumps(response))

    return RecommendationResponse(**response)
