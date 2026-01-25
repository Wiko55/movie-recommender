import logging
import os
import sys

import pandas as pd
from sqlalchemy.orm import Session

sys.path.append(os.getcwd())
from src import config, models
from src.database import SessionLocal, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def seed_data():
    db = SessionLocal()
    try:
        if db.query(models.Movie).first():
            logger.warning("Baza danych juz zawiera dane.")
            return
        logger.info("Rozpoczynanie procesu zasilania bazy danych...")
        logger.info(f"Wczytywanie danych filmow ze ścieki {config.MOVIES_PATH}")
        if not os.path.exists(config.MOVIES_PATH):
            logger.error(f"Plik {config.MOVIES_PATH} nie istnieje.")
            return
        movies_df = pd.read_csv(config.MOVIES_PATH)
        movies_objects = []
        for _, row in movies_df.iterrows():
            movies_objects.append(
                models.Movie(
                    id=int(row[config.COL_MOVIE_ID]),
                    title=row[config.COL_TITLE],
                    genres=row.get("genres", ""),
                )
            )
        db.add_all(movies_objects)
        db.commit()
        logger.info(f"Pomyślnie dodano {len(movies_objects)} filmów do bazy danych.")
        logger.info(f"Wczytywanie danych ocen ze ścieki {config.RATINGS_PATH}")
        if not os.path.exists(config.RATINGS_PATH):
            logger.error(f"Plik {config.RATINGS_PATH} nie istnieje.")
            return
        ratings_df = pd.read_csv(config.RATINGS_PATH)
        ratings_objects = []
        for _, row in ratings_df.iterrows():
            ratings_objects.append(
                models.Rating(
                    user_id=int(row[config.COL_USER_ID]),
                    movie_id=int(row[config.COL_MOVIE_ID]),
                    rating=float(row[config.COL_RATING]),
                    timestamp=int(row.get("timestamp", 0)),
                )
            )
        db.add_all(ratings_objects)
        db.commit()
        logger.info(f"Pomyślnie dodano {len(ratings_objects)} ocen do bazy danych.")
    except Exception as e:
        logger.error(f"Wystąpił błąd podczas zasilania bazy danych!", exc_info=True)
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    seed_data()
