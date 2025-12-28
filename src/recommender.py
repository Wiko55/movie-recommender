import datetime
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src import config

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod

from src.decorators import measure_execution_time


class BaseRecommender(ABC):
    """
    Abstrakcyjna klasa bazowa definiująca strukturę dla wszystkich rekomendatorów.
    """

    @abstractmethod
    def fit(self, matrix: pd.DataFrame) -> None:
        """Każdy model musi umieć się wytrenować."""
        pass

    @abstractmethod
    def recommend(self, user_id: int, top_n: int = 5) -> List[str]:
        """Każdy model musi umieć rekomendować."""
        pass


class MovieRecommender(BaseRecommender):
    """
    System rekomendacyjny oparty na klastrowaniu K-Means.
    Zapisuje stan modelu oraz pre-kalkulowane rekomendacje dla klastrów.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_recommendations: dict = {}
        self.user_cluster_map: dict = {}
        self.is_fitted: bool = False

    @measure_execution_time
    def fit(self, matrix: pd.DataFrame) -> None:
        """Trenuje model i liczy top filmy dla każdego klastra."""
        logger.info(f"Trenowanie K-Means dla {self.n_clusters} klastrów...")

        self.kmeans.fit(matrix)
        cluster_labels = self.kmeans.labels_

        logger.info("Pre-kalkulacja rekomendacji dla klastrów...")

        temp_df = matrix.copy()
        temp_df["cluster"] = cluster_labels

        self.user_cluster_map = dict(zip(matrix.index, cluster_labels))

        # Średnia ocen w klastrach
        for cluster_id in range(self.n_clusters):
            cluster_users = temp_df[temp_df["cluster"] == cluster_id]
            ratings_only = cluster_users.drop(columns=["cluster"])
            ratings_matrix = ratings_only.to_numpy()
            mean_ratings_array = np.mean(ratings_matrix, axis=0)
            mean_ratings = pd.Series(mean_ratings_array, index=ratings_only.columns)
            # mean_ratings = cluster_users.drop(columns=["cluster"]).mean(axis=0)
            top_movies = (
                mean_ratings.sort_values(ascending=False).head(20).index.tolist()
            )
            self.cluster_recommendations[cluster_id] = top_movies

        self.is_fitted = True
        logger.info("Model wytrenowany i zoptymalizowany.")

    def recommend(self, user_id: int, top_n: int = 5) -> List[str]:
        if not self.is_fitted:
            logger.error("Model nie jest wytrenowany!")
            return []

        cluster_id = self.user_cluster_map.get(user_id)

        if cluster_id is None:
            logger.warning(f"User {user_id} nieznany. Zwracam puste.")
            return []

        return self.cluster_recommendations.get(cluster_id, [])[:top_n]

    def save(self, path: Path) -> None:
        """Serializacja obiektu do pliku."""
        logger.info(f"Zapisywanie modelu do {path}")
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "MovieRecommender":
        """Deserializacja obiektu z pliku."""
        if not path.exists():
            raise FileNotFoundError(f"Brak modelu pod ścieżką: {path}")
        logger.info(f"Wczytywanie modelu z {path}")
        return joblib.load(path)
