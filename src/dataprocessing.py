import logging

import pandas as pd

from src import config  # Importowanie ustawień z config.py

# 1. Inicjalizacja loggera dla tego konkretnego pliku
logger = logging.getLogger(__name__)


def load_and_process() -> pd.DataFrame:
    """
    Proces ETL: Wczytuje surowe CSV, łączy je, filtruje i zwraca macierz.
    Pobiera ścieżki i nazwy kolumn z pliku config.py.
    """

    movies_path = config.MOVIES_PATH
    ratings_path = config.RATINGS_PATH

    if not movies_path.exists() or not ratings_path.exists():
        logger.error(f"KRYTYCZNY BŁĄD: Nie znaleziono plików w {config.DATA_DIR}")
        raise FileNotFoundError(
            f"Brakuje movies.csv lub ratings.csv w {config.DATA_DIR}"
        )

    logger.info("Rozpoczynanie procesu ETL (Extract-Transform-Load)...")

    try:
        df_movies = pd.read_csv(movies_path)
        df_ratings = pd.read_csv(ratings_path)

        df = pd.merge(df_ratings, df_movies, on=config.COL_MOVIE_ID)

        movie_stats = df.groupby(config.COL_TITLE)[config.COL_RATING].count()
        popular_movies = movie_stats[movie_stats > config.MIN_VOTES].index

        logger.info(
            f"Redukcja danych: {len(movie_stats)} -> {len(popular_movies)} filmów (Min. {config.MIN_VOTES} głosów)"
        )

        df_filtered = df[df[config.COL_TITLE].isin(popular_movies)]

        matrix = df_filtered.pivot_table(
            index=config.COL_USER_ID, columns=config.COL_TITLE, values=config.COL_RATING
        ).fillna(0)

        logger.info(f"ETL zakończony sukcesem. Rozmiar macierzy: {matrix.shape}")
        return matrix

    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania danych: {e}")
        raise e
