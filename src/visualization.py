import logging

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def save_cluster_plot(model, filename="klastry_filmow.png"):
    """
    Generuje wykres 2D klastrów używając PCA.
    Wymaganie: Wykres Matplotlib + biblioteka Sklearn (PCA).
    """
    print("Generowanie wykresu klastrów...")

    # Walidacja macierzy
    if not hasattr(model, "feature_matrix"):
        msg = "Model nie posiada 'feature_matrix'."
        logger.warning(msg)
        print(msg)
        return

    # Dane
    X = model.feature_matrix
    labels = model.kmeans.labels_

    # Redukcja wymiarów do 2D (oś X i Y)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis", alpha=0.6
    )

    plt.title("Wizualizacja Klastrów Filmowych (K-Means + PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="ID Klastra")
    plt.grid(True, alpha=0.3)

    # Zapis do pliku
    plt.savefig(filename)
    success_msg = f"Wykres zapisano pomyślnie jako: {filename}"
    logger.info(success_msg)
    print(success_msg)

    # Zamknięcie figury, żeby zwolnić pamięć
    plt.close()
