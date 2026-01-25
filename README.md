# 🎬 System Rekomendacji Filmów (Movie Recommender)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Coverage](https://img.shields.io/badge/Tests-Passing-brightgreen)

Projekt zaliczeniowy z przedmiotu **Język Python**. Aplikacja wykorzystuje uczenie maszynowe (algorytm K-Means Clustering) do grupowania użytkowników o podobnych gustach i rekomendowania im filmów.

---

* **Imię i nazwisko:** Wiktor Łach
* **Nr albumu:** 417725
* **Przedmiot:** Język Python (Rok 2025/2026)

---

## 📋 Realizacja Wymagań:

- [x] **Projekt uruchamiany z pliku `main.py`** (Tryb interaktywny CLI).
- [x] **Analiza danych:** Wykorzystanie `Pandas` i `NumPy` do przetwarzania macierzy ocen.
- [x] **Model ML:** Implementacja algorytmu K-Means z biblioteki `scikit-learn`.
- [x] **OOP i Wzorce:** Zastosowanie dziedziczenia (`BaseRecommender`) oraz wzorca **Strategia**.
- [x] **Wizualizacja:** Generowanie wykresu klastrów przy użyciu `Matplotlib` (PCA).
- [x] **Zaawansowany Python:** Własne dekoratory (`@measure_execution_time`), generatory (`yield` przy wczytywaniu) oraz Context Managery.
- [x] **Testy:** Testy jednostkowe (`pytest`) sprawdzające logikę i obliczenia.

---

## 🚀 Jak uruchomić projekt?

Projekt obsługuje dwa tryby działania: **CLI** oraz **Rozbudowany (Docker/API)**.

### 1. CLI
Uruchamia interaktywne menu w konsoli.

**Wymagania:** Python 3.10+ oraz zainstalowane zależności (`uv` lub `pip`).

```bash
# Instalacja zależności (przy używaniu uv)
uv sync

# Uruchomienie aplikacji
uv run python src/main.py

# Uruchomienie testów
uv run pytest tests/test_all.py
```

### Ewolucja projektu
Po wykonaniu wymagań projektowych dalej rozwijałem program. Zmieniłem bazę danych na PostgreSQL (przechowywuje oceny nowych użytkowników + hasła i loginy), dodałem opcję dotrenowania modelu "w locie" i wprowadziłem cache'owanie wyników rekomendacji (użyłem do tego Redis), dodałem opcję logowania oraz oceniania filmów oraz rozwiązałem problem cold startu, gdzie nowy użytkownik nie dostawał żadnych rekomendacji jeśli nie ocenił żadnych filmów. Natomiast program w tej formie nie spełniał wymagań projektu (cały projekt wstaje za pomocą konetenra w dockerze i jest obsługiwany za pomocą API). Cały ten kod o którym piszę znajduje się w branchu "testy".
