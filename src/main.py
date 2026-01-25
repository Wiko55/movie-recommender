import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.dataprocessing import load_and_process
from src.recommender import MovieRecommender
from src.schemas import HealthCheck, LandingPage, RecommendationResponse
from src.visualization import save_cluster_plot

# Setup loggera - wypisuje info w konsoli
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # czas
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("API")

# Globalna zmienna na modele
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Zarządzanie cyklem życia aplikacji.
    """
    import os

    password = os.getenv("Hasło")
    if password:
        logger.info("Hasło pomyśłnie ustawione")
    else:
        logger.info("Hasło nieustawione")
    # Ładowanie modelu
    try:
        model_path = Path("model_v1.joblib")

        if model_path.exists():
            logger.info(f"Ładowanie modelu z {model_path}...")
            ml_models["recommender"] = MovieRecommender.load(model_path)
            logger.info("Model załadowany do pamięci.")
        else:
            logger.error("❌ Brak pliku modelu!")
            ml_models["recommender"] = None
    except:
        logger.info("Nieudane ładowanie modelu")

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


@app.get("/recommend/{user_id}/{top_n}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, top_n: int):
    """
    Główny endpoint biznesowy.
    """
    model = ml_models.get("recommender")
    if not model:
        raise HTTPException(status_code=503, detail="Model nie jest gotowy.")

    recs = model.recommend(user_id, top_n=top_n)

    if not recs:
        logger.warning(f"Brak rekomendacji dla User {user_id}")

    return RecommendationResponse(
        user_id=user_id, recommendations=recs, model_version="v1"
    )


def run_cli_mode():
    """
    Funkcja uruchamiana TYLKO przez 'python src/main.py'.
    """

    print("\n" + "=" * 50)
    print("🎓 URUCHAMIANIE W TRYBIE CLI")
    print("=" * 50)

    # 1. Wczytywanie danych
    print("\n[KROK 1] Wczytywanie danych...")
    df = load_and_process()
    print(f"   -> Wczytano {len(df)} wierszy.")

    # 2. Trenowanie (Lokalna instancja, niezależna od API)
    print("\n[KROK 2] Inicjalizacja i trening modelu...")
    local_model = MovieRecommender()
    local_model.fit(df)

    max_id = df.index.max()  # Sprawdzenie ilu uzytkownikow jest w danych do trenowania
    while True:
        print("\n--- 📋 MENU GŁÓWNE ---")
        print("1. Generuj wykres klastrów")
        print("2. Generuj raport tekstowy")
        print("3. Uzyskaj rekomendację dla użytkownika")
        print("4. Wyjście")
        choice = input("\n Wybierz (1-4): ")

        if choice == "1":
            print("\nGenerowanie wykresu klastrów...")
            try:
                output_file = "wykres_rekomendacji.png"
                save_cluster_plot(local_model, output_file)
                print(f"   -> Sukces! Wykres zapisano jako '{output_file}'")
            except Exception as e:
                print(f"   -> Błąd generowania wykresu: {e}")
        elif choice == "2":
            local_model.generate_report("raport_wynikow.txt")

        elif choice == "3":
            while True:
                test_user_id = input(
                    f"Wprowadź dla jakiego id uzytkownika chcesz zobaczyć rekomendacje (zakres 1 do {max_id}): "
                )
                try:
                    test_user_id = int(test_user_id)
                    if (
                        1 <= test_user_id <= max_id or test_user_id not in local_model
                    ):  # użycie przeciążonego operatora in
                        break
                except:
                    print(f"Wprowadzono zły identyfikator, spróbuj ponownie")

            print(f"\n Rekomendacja dla User ID={test_user_id}:")
            try:
                recs = local_model[test_user_id]  # Użycie przeciążonego operatora []
                for i, movie in enumerate(recs, 1):
                    print(f"   {i}. {movie}")
            except Exception as e:
                print(f"   -> Błąd rekomendacji: {e}")

            print("\n✅ PROJEKT GOTOWY.")
            print("=" * 50 + "\n")
        elif choice == "4":
            print("Zamykanie programu")
            break
        else:
            print("Wybrana opcja jest niedostępna, wybierz opcję z zakresu 1-3")


if __name__ == "__main__":
    # Ten blok uruchamia się tylko, gdy uruchamia się stricte przez main
    run_cli_mode()
