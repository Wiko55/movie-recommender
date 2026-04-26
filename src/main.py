import glob
import json
import logging
import os
import pickle
import time
from contextlib import asynccontextmanager

import bcrypt
import pandas as pd
import redis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy import desc, func, text
from sqlalchemy.orm import Session

from src import models, schemas  # Import modeli, aby zarejestrować je w SQLAlchemy
from src.content_engine import ContentEngine
from src.data_loader import load_data
from src.database import SessionLocal, engine, get_db
from src.recommender import MovieRecommender
from src.schemas import RatingCreate
from src.tmdb_client import fetch_posters_for_movies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


ml_models = {}
redis_client = None
CACHE_MAX_ITEMS = 50
movies_data = None
links_data = None
ratings_data = None

content_engine = ContentEngine()
models.Base.metadata.create_all(bind=engine)


def get_password_hash(password):
    rpwd_bytes = password[:70].encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(rpwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password, hashed_password):
    rpwd_bytes = plain_password[:70].encode("utf-8")
    hashed_bytes = hashed_password.encode("utf-8")
    try:
        return bcrypt.checkpw(rpwd_bytes, hashed_bytes)
    except Exception as e:
        logger.error(f"Błąd podczas weryfikacji hasła: {e}")
        return False


def retrain_model_background(db_session: Session):
    """
    Funkcja do ponownego trenowania modelu w tle.
    """
    try:
        logger.info("🔄 Ponowne trenowanie modelu w tle...")
        sql_ratings = db_session.query(models.Rating).all()
        if not sql_ratings or len(sql_ratings) == 0:
            logger.info("Brak nowych ocen w SQL. Trenuję tylko na CSV.")
            return {"message": "Brak nowych ocen w SQL.", "status": "no_sql_data"}
        else:
            sql_df = pd.DataFrame(
                [
                    {"userId": r.user_id, "movieId": r.movie_id, "rating": r.rating}
                    for r in sql_ratings
                ]
            )
            logger.info(f"Pobrano {len(sql_df)} ocen z SQL.")

        combined_ratings = pd.concat([ratings_data, sql_df], ignore_index=True)

        recommender = MovieRecommender()
        recommender.fit(movies_data, combined_ratings)
        timestamp = int(time.time())
        filename = f"models/recommender_model_{timestamp}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(recommender, f)
        logger.info(f"💾 Nowy model zapisany do {filename}")
        ml_models["latest"] = recommender
        logger.info("✅ Model został pomyślnie ponownie wytrenowany.")
        if redis_client:
            redis_client.flushdb()
            logger.info("🗑️ Cache Redis został wyczyszczony po retrainingu.")
    except Exception as e:
        logger.error(f"❌ Błąd podczas ponownego trenowania modelu: {e}")
    finally:
        db_session.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Uruchamianie Systemu Rekomendacji")
    global redis_client, movies_data, links_data, ratings_data
    list_of_models = glob.glob("app/models/*.pkl")
    try:
        logger.info("📥 Pobieranie i ładowanie danych MovieLens...")
        movies, ratings, links = load_data()
        ratings_data = ratings
        movies_data = movies
        links_data = links
        max_id = ratings["userId"].max()
        start_id = round(max_id + 100, -2)
        logger.info(
            f"✅ Załadowano dane MovieLens: {len(movies)} filmów, {len(ratings)} ocen, max userId={max_id}"
        )
    except Exception as e:
        logger.error(f"❌ Błąd podczas ładowania danych MovieLens: {e}")
        start_id = 10000
    # Zmiana iterowania indexów
    try:
        db = SessionLocal()
        max_id = db.query(func.max(models.User.id))
        max_id = 0 if max_id is None else max_id
        start_id = max(start_id, max_id)
        # Komenda SQL, która przesuwa licznik
        db.execute(text(f"ALTER SEQUENCE users_id_seq RESTART WITH {start_id};"))
        db.commit()
        db.close()
        logger.info(f"🔧 Naprawiono sekwencję ID: Nowi userzy zaczną od ID {start_id}.")
    except Exception as e:
        logger.warning(
            f"⚠️ Nie udało się zaktualizować sekwencji ID (może to pierwsze uruchomienie?): {e}"
        )
    # --- ŁADOWANIE NOWEGO MODELU ---
    loaded_from_file = False
    if list_of_models:
        latest_model_file = max(list_of_models, key=os.path.getctime)
        with open(latest_model_file, "rb") as f:
            recommender = pickle.load(f)
            ml_models["latest"] = recommender
            loaded_from_file = True
            logger.info(f"✅ Załadowano model z pliku: {latest_model_file}")
    # --- JEŚLI BRAK MODELU W PLIKU, ŁADUJEMY Z DANYCH I GO ZAPISUJEMY---
    if not loaded_from_file:
        logger.info("⚠️ Brak modelu w plikach. Trenuję od zera...")
        try:
            logger.info("🧠 Trenowanie modelu k-NN...")
            recommender = MovieRecommender()
            recommender.fit(movies, ratings)
            ml_models["latest"] = recommender
            logger.info("✅ Model główny gotowy do pracy!")
            logger.info("📚 Trenowanie modelu Content-Based (TF-IDF)...")
            content_engine.fit(movies)
            logger.info("✅ Model Content-Based gotowy do pracy!")
            pickle.dump(recommender, open("models/recommender_model_initial.pkl", "wb"))
            logger.info("💾 Zapisano model początkowy do pliku.")
        except Exception as e:
            logger.error(f"❌ Błąd krytyczny podczas ładowania modelu: {e}")
            ml_models["latest"] = None

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

    yield

    ml_models.clear()
    if redis_client:
        redis_client.close()
    logger.info("🛑 Aplikacja zatrzymana.")


app = FastAPI(lifespan=lifespan, title="Movie Recommender API")
Instrumentator().instrument(app).expose(app)


class MovieItem(BaseModel):
    title: str
    poster: str | None = None
    movie_id: int | None = None


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[MovieItem]
    source: str
    model_version: str


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


@app.get("/", summary="Read Root")
def read_root():
    return {"message": "Witaj w API Rekomendacji Filmów (Wersja ML)"}


@app.get("/health", response_model=HealthCheck, summary="Health Check")
def health_check():
    return HealthCheck(
        status="ok",
        model_loaded=len(ml_models) > 0,
        redis_connected=redis_client is not None and redis_client.ping(),
    )


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_n: int = 5):
    """
    Zwraca rekomendacje na podstawie Collaborative Filtering.
    """
    limit = min(top_n, CACHE_MAX_ITEMS)
    cache_key = f"rec_img_v1_{user_id}"

    try:
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return RecommendationResponse(**data)
    except Exception as e:
        logger.error(f"Redis error: {e}")

    model = ml_models.get("latest")
    if not model:
        raise HTTPException(status_code=503, detail="Model ML nie jest gotowy")

    try:
        titles = model.recommend(user_id, top_n=CACHE_MAX_ITEMS)
    except Exception as e:
        logger.error(f"Błąd modelu: {e}")

    if not titles:
        logger.info(f"Cold Start dla User {user_id} - serwuję Hity Globalne")
        titles = model.get_popular_movies(top_n=CACHE_MAX_ITEMS)
        logger.info(f" {user_id} - serwuję Hity Globalne")
        source = "popularity_fallback"

    if titles:
        final_items = fetch_posters_for_movies(
            movies_data[movies_data["title"].isin(titles)], links_data, top_n=limit
        )

    if redis_client and final_items:
        payload = {
            "user_id": user_id,
            "recommendations": final_items,
            "source": "model_computation",
            "model_version": "v2_knn",
        }
        try:
            redis_client.setex(
                cache_key,
                60 * 10,
                json.dumps(payload),  # na 10 minut cache ustawiony
            )
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
    cache_key = f"rec_movie_{movie_title.replace(' ', '_').lower()}"

    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            return RecommendationResponse(**json.loads(cached))

    model = ml_models.get("latest")
    if not model or movies_data is None:
        raise HTTPException(status_code=503, detail="System niegotowy")

    titles = model.get_recommendations_for_movie(movie_title, top_n=limit)

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

    if redis_client:
        redis_client.setex(cache_key, 600, json.dumps(response))

    return RecommendationResponse(**response)


@app.get("/movies", response_model=list[schemas.Movie])
def get_movies(
    skip: int = 0,
    limit: int = 10,
    search: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(models.Movie)

    if search:
        query = query.filter(models.Movie.title.contains(search))

    movies = query.offset(skip).limit(limit).all()
    return movies


@app.get("/recommendations/top", response_model=list[schemas.Movie])
def get_top_rated_movies(limit: int = 5, db: Session = Depends(get_db)):
    """
    Zwraca listę top N najwyżej ocenianych filmów.
    """
    top_movies = (
        db.query(models.Movie)
        .join(models.Rating, models.Movie.id == models.Rating.movie_id)
        .group_by(models.Movie.id)
        .order_by(func.avg(models.Rating.rating).desc())
        .limit(limit)
        .all()
    )
    return top_movies


@app.get("/users/{user_id}", response_model=list[schemas.Movie])
def get_user_watch_history(user_id: int, limit: int = 3, db: Session = Depends(get_db)):
    """
    Zwraca ulubione filmy użytkownika na podstawie jego ocen.
    """
    watched_movies = (
        db.query(models.Movie)
        .join(models.Rating, models.Movie.id == models.Rating.movie_id)
        .where(models.Rating.user_id == user_id)
        .order_by(models.Rating.rating.desc())
        .limit(limit)
        .all()
    )
    return watched_movies


@app.post("/ratings/", summary="Rate a movie")
def rate_movie(rating: RatingCreate, db: Session = Depends(get_db)):
    """
    Dodaje ocenę filmu przez użytkownika, jeśli istnieje to UPDATE jeśli istnieje to INSERT
    """
    if rating.rating < 0 or rating.rating > 5:
        raise HTTPException(status_code=400, detail="Ocena musi być w zakresie 0-5")
    movie = db.query(models.Movie).filter(models.Movie.id == rating.movie_id).first()
    if not movie:
        if movies_data is not None and rating.movie_id in movies_data["movieId"].values:
            title = movies_data[movies_data["movieId"] == rating.movie_id].iloc[0][
                "title"
            ]
            new_movie_obj = models.Movie(
                id=rating.movie_id, title=title, genres="Unknown"
            )
            db.add(new_movie_obj)
            db.commit()
            logger.info(
                f"🔄 Auto-migracja: Dodano film ID {rating.movie_id} z CSV do SQL."
            )
        else:
            raise HTTPException(
                status_code=404, detail="Film nie znaleziony w bazie ani w plikach."
            )
    rating_db = (
        db.query(models.Rating)
        .filter(
            models.Rating.user_id == rating.user_id,
            models.Rating.movie_id == rating.movie_id,
        )
        .first()
    )
    if rating_db:
        rating_db.rating = rating.rating
        logger.info(
            f"Użytkownik {rating.user_id} zaktualizował ocenę filmu z {rating.movie_id} na {rating.rating}"
        )
    else:
        new_rating = models.Rating(
            user_id=rating.user_id,
            movie_id=rating.movie_id,
            rating=rating.rating,
            timestamp=0,
        )
        db.add(new_rating)
        logger.info(
            f"Użytkownik {rating.user_id} dodał ocenę {rating.rating} dla filmu {rating.movie_id}"
        )
    db.commit()

    if redis_client:
        cache_key = f"rec_img_v1_{rating.user_id}"
        redis_client.delete(cache_key)
    return {"message": "Ocena zapisana"}


@app.get("/admin/stats")
def get_admin_stats(db: Session = Depends(get_db)):
    """
    Zwraca statystyki łączone: Baza SQL (nowe) + Pliki CSV (stare MovieLens).
    """
    try:
        sql_ratings_count = db.query(models.Rating).count()
        sql_users_count = db.query(models.User).count()

        csv_ratings_count = len(ratings_data) if ratings_data is not None else 0
        csv_users_count = (
            ratings_data["userId"].nunique() if ratings_data is not None else 0
        )

        top_movies = []
        if ratings_data is not None and movies_data is not None:
            popular = ratings_data.groupby("movieId").size().reset_index(name="counts")
            popular = popular.sort_values("counts", ascending=False).head(5)

            merged = popular.merge(movies_data, on="movieId")

            for _, row in merged.iterrows():
                top_movies.append(
                    {
                        "title": row["title"],
                        "count": int(row["counts"]),
                    }
                )

        return {
            "total_ratings": sql_ratings_count + csv_ratings_count,  # Suma
            "active_users": sql_users_count + csv_users_count,  # Suma
            "top_movies": top_movies,
        }

    except Exception as e:
        logger.error(f"Błąd statystyk: {e}")
        return {"total_ratings": 0, "active_users": 0, "top_movies": []}


@app.get("/recommendations/content/{title}")
def recommend_by_content(title: str, top_n: int = 5):
    """
    Rekomdenduje filmy z tego samego gatunku na podstawie tytułu filmu.
    """
    clean_title = title.strip().lower().replace(" ", "_")
    cache_key = f"content_rec_{clean_title}_{top_n}"
    if redis_client:
        cached = redis_client.get(cache_key)
        if cached:
            logger.info(f"Zwracanie z cache content-based dla klucza: {cache_key}")
            return json.loads(cached)
    try:
        titles = content_engine.recommend(title, top_n=top_n)
        logger.info(f"Rekomendacje content-based dla '{title}': {titles}")
        if not titles:
            logger.info(f"Brak rekomendacji content-based dla tytułu: {title}")
            return {"message": "Brak rekomendacji content-based dla podanego tytułu."}
        final_items = fetch_posters_for_movies(
            movies_data[movies_data["title"].isin(titles)], links_data, top_n=top_n
        )
        if redis_client:
            payload = {
                "searched_movie": title,
                "recommendations": final_items,
                "source": "TF-IDF Content-Based",
            }
            redis_client.setex(cache_key, 3600, json.dumps(payload))
        return payload
    except Exception as e:
        logger.error(f"Błąd podczas rekomendacji content-based: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Błąd serwera podczas rekomendacji content-based. {e}",
        )


# DOTRENOWYWANIE MODELU
@app.post("/admin/retrain", summary="Retrain ML Model in Background")
def retrain_model_endpoint(
    background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """
    Endpoint do ponownego trenowania modelu w tle.
    """
    background_tasks.add_task(retrain_model_background, db_session=db)
    return {"message": "Ponowne trenowanie modelu zostało uruchomione w tle."}


@app.post("/register", summary="Register a new user")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Rejestruje nowego użytkownika o ile nie istnieje już taki w bazie.
    """
    existing_user = (
        db.query(models.User).filter(models.User.name == user.username).first()
    )
    if existing_user:
        logger.warning(f"Próba rejestracji istniejącego użytkownika: {user.username}")
        raise HTTPException(status_code=400, detail="Nazwa użytkownika już istnieje")

    hashed_password = get_password_hash(user.password)
    new_user = models.User(name=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"Zarejestrowano nowego użytkownika: {user.username}")
    return {"message": "Użytkownik zarejestrowany pomyślnie", "user_id": new_user.id}


@app.post("/login", summary="User login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    """
    Loguje użytkownika jeśli dane są poprawne.
    """
    db_user = db.query(models.User).filter(models.User.name == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        logger.warning(f"Nieudana próba logowania użytkownika: {user.username}")
        raise HTTPException(
            status_code=401, detail="Nieprawidłowa nazwa użytkownika lub hasło"
        )

    logger.info(f"Użytkownik zalogowany: {user.username}")
    return {
        "message": "Zalogowano pomyślnie",
        "user_id": db_user.id,
        "username": db_user.name,
    }
