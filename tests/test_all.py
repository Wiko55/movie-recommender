import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src.dataprocessing import load_and_process
from src.recommender import MovieRecommender


# Test 1: Czy dane ładują się poprawnie
def test_data_loading():
    df = load_and_process()
    print(df.columns)
    assert not df.empty, "Ramka danych jest pusta"
    assert df.index.name == "userId", "Indeksem tabeli powinno być userId"


# Test 2: Czy model poprawnie zapisuje macierz (potrzebną do wykresu)
def test_model_training_and_matrix():
    df = load_and_process()
    df_sample = df.head(20)

    rec = MovieRecommender(n_clusters=2)
    rec.fit(df_sample)

    assert rec.feature_matrix is not None, "Macierz cech nie została zapisana"
    assert len(rec.feature_matrix) == 20, "Wymiary macierzy się nie zgadzają"


# Test 3: Czy rekomendacja zwraca listę napisów
def test_recommendation_format():
    df = load_and_process()
    rec = MovieRecommender()
    rec.fit(df.head(50))

    recommendations = rec.recommend(user_id=1, top_n=3)

    assert isinstance(recommendations, list), "Wynik powinien być listą"
    assert len(recommendations) > 0, "Lista rekomendacji jest pusta"
    assert isinstance(recommendations[0], str), "Elementy listy powinny być tytułami"
