from pathlib import Path

from src import config
from src.dataprocessing import load_and_process
from src.recommender import MovieRecommender

# Upewnij się, że masz dane w data/raw!
if config.MOVIES_PATH.exists():
    df = load_and_process()
    rec = MovieRecommender()
    rec.fit(df)
    rec.save(Path("model_v1.joblib"))
    print("Model zapisany.")
else:
    print("Brak danych CSV!")
