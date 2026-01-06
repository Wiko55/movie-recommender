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

        # 1. Pivot Table: Wiersze=Filmy, Kolumny=Userzy, Wartości=Oceny
        # Wypełniamy zera (0) tam, gdzie ktoś nie ocenił filmu
        pivot_table = ratings.pivot(
            index="movieId", columns="userId", values="rating"
        ).fillna(0)

        # 2. Konwersja do macierzy rzadkiej (Sparse Matrix) - oszczędza RAM
        # Nie trzymamy milionów zer w pamięci, tylko współrzędne ocen.
        self.movie_user_mat = csr_matrix(pivot_table.values)

        # Mapa mapująca ID filmu na indeks w macierzy (bo mogą być dziury w ID)
        self.movie_to_idx = {movie: i for i, movie in enumerate(pivot_table.index)}
        self.idx_to_movie = {i: movie for i, movie in enumerate(pivot_table.index)}

        # 3. Trening modelu (to trwa ułamek sekundy na małym zbiorze)
        self.model.fit(self.movie_user_mat)
        print(
            "✅ Model wytrenowany na macierzy o wymiarach:", self.movie_user_mat.shape
        )

    def recommend(self, user_id: int, top_n: int = 5):
        """
        Prosta logika rekomendacji dla usera:
        1. Znajdź filmy, które user ocenił najlepiej (5.0).
        2. Dla każdego ulubionego filmu znajdź "sąsiadów" (podobne filmy).
        """
        # Pobierz filmy ocenione przez tego usera
        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]

        if user_ratings.empty:
            return []

        # Weź filmy ocenione na >= 4.0, posortowane malejąco
        favorites = user_ratings[user_ratings["rating"] >= 4.0].sort_values(
            "rating", ascending=False
        )

        if favorites.empty:
            # Jak ktoś tylko hejtuje filmy, weź cokolwiek co widział
            favorites = user_ratings.sort_values("rating", ascending=False).head(3)

        # Bierzemy top 3 ulubione filmy usera i szukamy dla nich podobnych
        recommendations = []

        for movie_id in favorites["movieId"].head(3).tolist():
            if movie_id not in self.movie_to_idx:
                continue

            # Znajdź indeks filmu w macierzy
            idx = self.movie_to_idx[movie_id]

            # Zapytaj model o sąsiadów
            distances, indices = self.model.kneighbors(
                self.movie_user_mat[idx], n_neighbors=top_n + 1
            )

            # Dodaj znalezione tytuły do zbioru (zbiór usuwa duplikaty)
            for i, d in zip(
                indices.flatten()[1:top_n], distances.flatten()[1:top_n]
            ):  # Pomin [0], bo to ten sam film
                neighbor_id = self.idx_to_movie[i]
                # Pobierz tytuł z DataFrame'a filmów
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

        return [final_list]

    def get_popular_movies(self, top_n: int = 5):
        """
        Zwraca listę tytułów najpopularniejszych filmów (wg liczby ocen).
        """
        if self.ratings_df is None or self.movies_df is None:
            return []

        # 1. Liczymy ile razy każdy film był oceniany
        rating_counts = self.ratings_df.groupby("movieId").count()["rating"]

        # 2. Sortujemy malejąco i bierzemy top N ID filmów
        top_ids = rating_counts.sort_values(ascending=False).head(top_n).index

        # 3. Zamieniamy ID na tytuły
        popular_titles = self.ratings_df[self.ratings_df["movieId"].isin(top_ids)][
            "title"
        ].tolist()

        return popular_titles[:top_n]

    def get_recommendations_for_movie(self, movie_title: str, top_n: int = 5):
        """
        Znajduje filmy podobne do podanego tytułu (Item-Item Filtering).
        """
        # 1. Znajdź ID filmu na podstawie tytułu (szukanie fragmentu tekstu)
        # Robimy lowercase, żeby wielkość liter nie miała znaczenia
        matches = self.movies_df[
            self.movies_df["title"].str.contains(movie_title, case=False, na=False)
        ]

        if matches.empty:
            return []

        # Bierzemy pierwszy pasujący film (najbardziej prawdopodobny)
        target_movie_id = matches.iloc[0]["movieId"]
        target_title = matches.iloc[0]["title"]
        print(f"🔍 Szukam podobnych do: {target_title} (ID: {target_movie_id})")

        if target_movie_id not in self.movie_to_idx:
            return []

        # 2. Pobierz wektor tego filmu
        idx = self.movie_to_idx[target_movie_id]
        movie_vec = self.movie_user_mat[idx]
        print(f"******** CSR: *******\n {self.movie_user_mat}")
        print(f"******** CSR[idx]: *******\n {movie_vec}")
        # 3. Znajdź sąsiadów
        distances, indices = self.model.kneighbors(movie_vec, n_neighbors=top_n + 1)

        recommendations = []
        for i in indices.flatten():
            neighbor_id = self.idx_to_movie[i]
            if neighbor_id == target_movie_id:
                continue  # Pomiń ten sam film

            title = self.movies_df[self.movies_df["movieId"] == neighbor_id][
                "title"
            ].values[0]
            recommendations.append(title)

        return recommendations[:top_n]


if __name__ == "__main__":
    # Szybki test lokalny
    from data_loader import load_data

    m, r = load_data()

    rec = MovieRecommender()
    rec.fit(m, r)

    print("\nRekomendacje dla Usera 1:")
    print(rec.recommend(1, 5))
