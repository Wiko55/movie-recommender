import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentEngine:
    def __init__(self):
        self.tfidf_matrix = None
        self.movies_df = None
        self.indices = None

    def fit(self, movies: pd.DataFrame):
        """
        Uczy się treści filmów - gatunków
        """
        self.movies_df = movies.reset_index(drop=True)
        self.movies_df["genres_str"] = self.movies_df["genres"].replace("|", " ")
        tf_idf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tf_idf.fit_transform(self.movies_df["genres_str"])
        logger.info(
            f"Model ITF-IDF został wytrenowany. Wymiary macierzy: {self.tfidf_matrix.shape}"
        )
        self.indices = pd.Series(
            self.movies_df.index, index=self.movies_df["movieId"]
        ).drop_duplicates()

    def recommend(self, title: str, top_n: int = 5):
        """
        Rekomenduje filmy na podstawie tytułu
        """
        if title in self.movies_df["title"].values:
            idx = self.indices[title]
        else:
            matches = self.movies_df[
                self.movies_df["title"].str.contains(
                    title, case=False, na=False, regex=False
                )
            ]
            if len(matches) == 0:
                logger.info(f"Brak filmu o tytule zawierającym: {title}")
                return []
            title_sim = matches.iloc[0]["title"]
            idx = self.indices[
                self.movies_df[self.movies_df["title"] == title_sim]["movieId"]
            ].iloc[0]

            logger.info(
                f'Znaleziono film o tytule podobnym do "{title}": "{title_sim}"'
            )

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        cosine_similarities = linear_kernel(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()
        sim_scores = list(enumerate(cosine_similarities))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : top_n + 1]
        movie_indices = [i[0] for i in sim_scores]

        return self.movies_df["title"].iloc[movie_indices].tolist()
