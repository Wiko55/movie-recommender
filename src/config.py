from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"

COL_USER_ID = "userId"
COL_MOVIE_ID = "movieId"
COL_TITLE = "title"
COL_RATING = "rating"

MIN_VOTES = 20
