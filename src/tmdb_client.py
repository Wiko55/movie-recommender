import os

import pandas as pd
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def get_poster_url(tmdb_id):
    """
    Pobiera URL plakatu dla danego tmdb_id.
    """
    if not TMDB_API_KEY:
        return None

    try:
        url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"{POSTER_BASE_URL}{poster_path}"
    except Exception:
        pass  # Ignorujemy błędy sieciowe, po prostu nie będzie obrazka
    return None


def fetch_posters_for_movies(movies_df: pd.DataFrame, links_df: pd.DataFrame, top_n=5):
    """
    Dla listy filmów znajduje ich plakaty.
    """
    # Łączymy filmy z ich ID w bazie TMDB (plik links.csv)
    movies_with_links = movies_df.merge(links_df, on="movieId", how="left")

    results = []
    for _, row in movies_with_links.head(top_n).iterrows():
        title = row["title"]
        tmdb_id = row["tmdbId"]

        poster = None
        # tmdbId może być NaN (brak danych)
        if pd.notna(tmdb_id):
            poster = get_poster_url(int(tmdb_id))

        results.append({"title": title, "poster": poster})

    return results
