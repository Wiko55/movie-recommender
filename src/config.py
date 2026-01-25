import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

COL_USER_ID = "userId"
COL_MOVIE_ID = "movieId"
COL_TITLE = "title"
COL_RATING = "rating"

DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "db")
DB_USER = os.getenv("DB_USER", "my_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret_password")
DB_PORT = int(os.getenv("DB_PORT", 5432))

MIN_VOTES = int(os.getenv("MIN_VOTES", "20"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "20"))
