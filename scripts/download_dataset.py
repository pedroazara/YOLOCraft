from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

DATASET = "pierreayfri/minecraft-mobs"

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "minecraft_mobs-2"
ZIP_FILE = DATA_DIR / "minecraft-mobs-2.zip"


def download_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"Dataset already exists at {DATASET_DIR.resolve()}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset...")
    api.dataset_download_files(
        DATASET,
        path=DATA_DIR,
        unzip=False
    )

    print("Extracting files...")

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    ZIP_FILE.unlink()

    print(f"Dataset saved at {DATASET_DIR.resolve()}")


if __name__ == "__main__":
    download_dataset()
