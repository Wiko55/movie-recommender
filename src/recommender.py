import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(self):
        self.model = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1
        )
        self.movie_user_mat = None
        self.movies_df = None
        self.ratings_df = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame):
        """
        Trenuje model: Tworzy macierz Film-Użytkownik i uczy k-NN.
        """
        self.movies_df = movies
        self.ratings_df = ratings
        pivot_table = ratings.pivot(
            index="movieId", columns="userId", values="rating"
        ).fillna(0)

        self.movie_user_mat = csr_matrix(pivot_table.values)

        self.movie_to_idx = {movie: i for i, movie in enumerate(pivot_table.index)}
        self.idx_to_movie = {i: movie for i, movie in enumerate(pivot_table.index)}

        self.model.fit(self.movie_user_mat)
        print("Model wytrenowany na macierzy o wymiarach:", self.movie_user_mat.shape)

    def recommend(self, user_id: int, top_n: int = 5):
        """
        Prosta logika rekomendacji dla usera:
        1. Znajdź filmy, które user ocenił najlepiej (5.0).
        2. Dla każdego ulubionego filmu znajdź "sąsiadów" (podobne filmy).
        """
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]

        if user_ratings.empty:
            return []

        favorites = user_ratings[user_ratings["rating"] >= 4.0].sort_values(
            "rating", ascending=False
        )

        if favorites.empty:  # jeśli użytownik nie ocenia filmów wysoko to top3 globalne
            favorites = user_ratings.sort_values("rating", ascending=False).head(3)

        recommendations = []

        for movie_id in favorites["movieId"].head(3).tolist():
            if movie_id not in self.movie_to_idx:
                continue

            idx = self.movie_to_idx[movie_id]

            distances, indices = self.model.kneighbors(
                self.movie_user_mat[idx], n_neighbors=top_n + 1
            )

            for i, d in zip(
                indices.flatten()[1:top_n],
                distances.flatten()[
                    1:top_n
                ],  # indexowanie od 1, bo pierwszy to ten sam film
            ):
                neighbor_id = self.idx_to_movie[i]
                title = self.movies_df[self.movies_df["movieId"] == neighbor_id][
                    "title"
                ].values[0]
                scoring = (1 - d) * user_ratings[user_ratings["movieId"] == movie_id][
                    "rating"
                ].values[0]
                recommendations.append([scoring, title])
        movies_sorted = sorted(
            list(recommendations), reverse=True, key=lambda item: item[0]
        )[:top_n]
        seen_titles = set()
        final_list = []

        for score, title in movies_sorted:
            if title not in seen_titles:
                final_list.append(title)
                seen_titles.add(title)
            if len(final_list) >= top_n:
                break

        return final_list

    def get_popular_movies(self, top_n: int = 5):
        """
        Zwraca listę tytułów najpopularniejszych filmów (wg liczby ocen).
        """
        if self.ratings_df is None or self.movies_df is None:
            return []

        rating_counts = self.ratings_df.groupby("movieId").count()["rating"]
        top_ids = rating_counts.sort_values(ascending=False).head(top_n).index

        print(f"test  {self.ratings_df.head()}")
        print(f"test movies {self.movies_df.head()['title']}")
        popular_titles = self.movies_df[self.movies_df["movieId"].isin(top_ids)][
            "title"
        ].tolist()

        return popular_titles[:top_n]

    def get_recommendations_for_movie(self, movie_title: str, top_n: int = 5):
        """
        Znajduje filmy podobne do podanego tytułu (Item-Item Filtering).
        """
        matches = self.movies_df[
            self.movies_df["title"].str.contains(movie_title, case=False, na=False)
        ]

        if matches.empty:
            return []

        target_movie_id = matches.iloc[0]["movieId"]
        target_title = matches.iloc[0]["title"]
        print(f"🔍 Szukam podobnych do: {target_title} (ID: {target_movie_id})")

        if target_movie_id not in self.movie_to_idx:
            return []

        idx = self.movie_to_idx[target_movie_id]
        movie_vec = self.movie_user_mat[idx]
        distances, indices = self.model.kneighbors(movie_vec, n_neighbors=top_n + 1)

        recommendations = []
        for i in indices.flatten():
            neighbor_id = self.idx_to_movie[i]
            if neighbor_id == target_movie_id:
                continue  # pominięcie tego samego filmu

            title = self.movies_df[self.movies_df["movieId"] == neighbor_id][
                "title"
            ].values[0]
            recommendations.append(title)

        return recommendations[:top_n]


if __name__ == "__main__":  # do testów lokalnych
    from data_loader import load_data

    m, r = load_data()

    rec = MovieRecommender()
    rec.fit(m, r)

    print("\nRekomendacje dla Usera 1:")
    print(rec.recommend(1, 5))
