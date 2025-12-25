import shutil
from pathlib import Path

import kagglehub


def download_movielens():
    print("⏳ Pobieranie MovieLens Dataset...")
    path = kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset")
    print(f" Pobrano do tymczasowego folderu: {path}")
    source_dir = Path(path)
    destination_dir = Path(__file__).parent.parent / "data" / "raw"
    destination_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = ["ratings.csv", "movies.csv"]

    for file_name in files_to_copy:
        found = list(source_dir.rglob(file_name))
        if found:
            source_file = found[0]
            dest_file = destination_dir / file_name
            shutil.copy(source_file, dest_file)
        else:
            print(f"Nie znaleziono pliku {file_name}")


if __name__ == "__main__":
    download_movielens()
