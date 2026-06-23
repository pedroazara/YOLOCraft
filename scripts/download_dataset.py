# download_dataset.py

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

DATASET = "dracotlw/minecraft-mobs-yolo-dataset"

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "minecraft_mobs"
ZIP_FILE = DATA_DIR / "minecraft-mobs-yolo-dataset.zip"


def download_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"✓ Dataset já existe em {DATASET_DIR.resolve()}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Autenticando no Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print("Baixando dataset...")
    api.dataset_download_files(
        DATASET,
        path=DATA_DIR,
        unzip=False
    )

    print("Extraindo arquivos...")

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    ZIP_FILE.unlink()

    print(f"✓ Dataset salvo em {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    download_dataset()