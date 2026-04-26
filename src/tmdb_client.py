import logging
import os

import requests

logger = logging.getLogger(__name__)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


def get_poster_url(tmdb_id):
    """
    Pobiera link plakatu
    """
    if not TMDB_API_KEY:
        logger.warning("Brak klucza TMDB_API_KEY w ENV")
        return None

    if not tmdb_id or str(tmdb_id) == "nan":
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
    logger.info(f"Pobieranie plakatu z TMDB, link {url}")
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception as e:
        logger.error(f"Błąd połączenia z TMDB dla ID {tmdb_id}: {e}")

    return None


def fetch_posters_for_movies(movies_df, links_df, top_n=5):
    """
    Uzupełnia grafiki na filmach
    """
    results = []

    for _, row in movies_df.head(top_n).iterrows():
        movie_id = row["movieId"]
        title = row["title"]

        tmdb_id = None
        if links_df is not None:
            link_row = links_df[links_df["movieId"] == movie_id]
            if not link_row.empty:
                tmdb_id = link_row.iloc[0]["tmdbId"]

        poster_url = get_poster_url(tmdb_id)

        results.append(
            {
                "movie_id": int(movie_id),
                "title": title,
                "poster": poster_url,
            }
        )

    return results
