import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.recommender import MovieRecommender
from src.schemas import HealthCheck, LandingPage, RecommendationResponse

# Setup loggera
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Globalna zmienna na modele
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Zarządzanie cyklem życia aplikacji.
    """
    model_path = Path("model_v1.joblib")

    if model_path.exists():
        logger.info(f"Ładowanie modelu z {model_path}...")
        ml_models["recommender"] = MovieRecommender.load(model_path)
        logger.info("Model załadowany do pamięci.")
    else:
        logger.error("❌ Brak pliku modelu!")
        ml_models["recommender"] = None

    yield

    ml_models.clear()
    logger.info("Aplikacja zatrzymana.")


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


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_n: int = 5):
    """
    Główny endpoint biznesowy.
    """
    model = ml_models.get("recommender")
    if not model:
        raise HTTPException(status_code=503, detail="Model nie jest gotowy.")

    # Wywołanie logiki biznesowej
    recs = model.recommend(user_id, top_n=top_n)

    if not recs:
        logger.warning(f"Brak rekomendacji dla User {user_id}")

    return RecommendationResponse(
        user_id=user_id, recommendations=recs, model_version="v1"
    )
