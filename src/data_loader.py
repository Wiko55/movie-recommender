import io
import os
import zipfile

import pandas as pd
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
LINKS_PATH = os.path.join(DATA_DIR, "links.csv")


def download_and_extract_data():
    """Pobiera i rozpakowuje dataset MovieLens, jeśli nie istnieje."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(MOVIES_PATH) or not os.path.exists(RATINGS_PATH):
        print("Pobieranie danych MovieLens...")
        r = requests.get(MOVIELENS_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))

        # Wyciągamy pliki z podkatalogu ml-latest-small do naszego katalogu data
        for file in z.namelist():
            if file.endswith("movies.csv"):
                with open(MOVIES_PATH, "wb") as f:
                    f.write(z.read(file))
            elif file.endswith("ratings.csv"):
                with open(RATINGS_PATH, "wb") as f:
                    f.write(z.read(file))
            elif file.endswith("links"):
                with open(LINKS_PATH, "wb") as f:
                    f.write(z.read(file))
        print("Dane pobrane i zapisane w folderze data/")
    else:
        print("Dane już istnieją.")


def load_data():
    """Ładuje dane do DataFrame'ów."""
    download_and_extract_data()

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    links = pd.read_csv(LINKS_PATH)

    return movies, ratings, links


if __name__ == "__main__":
    # Test ręczny
    m, r = load_data()
    print(f"Filmy: {m.shape}, Oceny: {r.shape}")
    print(m.head())
    print(r["rating"].mean())
