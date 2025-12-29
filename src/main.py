import logging
from contextlib import asynccontextmanager
from pathlib import Path

import redis
from fastapi import FastAPI, HTTPException
from src.visualization import save_cluster_plot

from src import config
from src.dataprocessing import load_and_process
from src.recommender import MovieRecommender
from src.schemas import HealthCheck, LandingPage, RecommendationResponse

# Setup loggera - wypisuje info w konsoli
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Setup redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = None
# Cache
CACHE_MAX_ITEMS = config.CACHE_MAX_ITEMS

# Globalna zmienna na modele
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Zarządzanie cyklem życia aplikacji.
    """
    # Uruchamianie Redisa
    global redis_client
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST, port=6379, db=0, decode_responses=True
        )
        redis_client.ping()  # Test połączenia
        logger.info(f"Połączono z Redisem, host: {REDIS_HOST}")
    except redis.ConnectionError:
        logger.warning("Nie można połączyć z Redisem, cache nie będzie działać.")
        redis_client = None

    # Ładowanie modelu
    try:
        model_path = Path("model_v1.joblib")

        if model_path.exists():
            logger.info(f"Ładowanie modelu z {model_path}...")
            ml_models["recommender"] = MovieRecommender.load(model_path)
            logger.info("Model załadowany do pamięci.")
        else:
            logger.error("❌ Brak pliku modelu!")
            ml_models["recommender"] = None
    except:
        logger.info("Nieudane ładowanie modelu")

    yield

    ml_models.clear()
    if redis_client:
        redis_client.close()
    logger.info("Aplikacja zatrzymana, Redis zamknięty")


# Inicjalizacja FastAPI z defined lifespan
app = FastAPI(title="Movie Recommender API", lifespan=lifespan)


@app.get("/")
async def read_root():
    return LandingPage(status_serwera="Żyje", messege="Strona z rekomendacjami filmów")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Endpoint dla Kubernetesa?"""
    is_ready = ml_models.get("recommender") is not None
    return HealthCheck(status="ok", running_model=is_ready)


@app.get("/recommend/{user_id}/{top_n}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_n: int):
    """
    Główny endpoint biznesowy.
    """
    top_n = CACHE_MAX_ITEMS if top_n < CACHE_MAX_ITEMS else top_n
    # 1. Sprawdzenie cache
    cache_key = f"rec_user_{user_id}"
    if redis_client:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            data = json.loads(cached_data)
            logger.info(f"CACHE HIT: Mam {CACHE_MAX_ITEMS} filmów, zwracam {top_n}")

            data["recommendations"] = data["recommendations"][:top_n]
            data["source"] = "cache"  # Nadpisujemy źródło
            return RecommendationResponse(
                user_id=user_id,
                recommendations=data["recommendations"][:top_n],
                source="cache",
                model_version="v1",
            )

    # 2. OBLICZENIA (Cache Miss)
    logger.info(f"CACHE MISS: Liczę model dla {user_id}")
    model = ml_models.get("recommender")

    if model is None:
        raise HTTPException(status_code=503, detail="Model niedostępny")

    try:
        full_recs = model.recommend(user_id, top_n=CACHE_MAX_ITEMS)

        response_payload = {
            "user_id": user_id,
            "source": "model_computation",
            "recommendations": full_recs,  # Tutaj zapisujemy całą 50-tkę
        }

        # 3. ZAPIS DO CACHE (Pełna lista 50 filmów)
        if redis_client:
            redis_client.setex(cache_key, 60, json.dumps(response_payload))

        return RecommendationResponse(
            user_id=user_id,
            recommendations=full_recs[:top_n],
            source="model_computation",
            model_version="v1",
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
