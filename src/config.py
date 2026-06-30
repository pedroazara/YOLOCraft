from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DATASETS = {
    "minecraft_mobs": DATA_DIR / "minecraft_mobs" / "minecraft_mobs_yolo",
    "minecraft_mobs-2": DATA_DIR / "minecraft_mobs-2" / "yolo",
}

DEFAULT_DATASET = "minecraft_mobs"


def get_dataset(name=DEFAULT_DATASET):
    dataset_dir = DATASETS[name]
    return {
        "dir": dataset_dir,
        "train_images": dataset_dir / "train" / "images",
        "train_labels": dataset_dir / "train" / "labels",
        "val_images": dataset_dir / "val" / "images",
        "val_labels": dataset_dir / "val" / "labels",
        "test_images": dataset_dir / "test" / "images",
        "test_labels": dataset_dir / "test" / "labels",
        "data_yaml": dataset_dir / "data.yaml",
    }


DATASET_DIR = get_dataset()["dir"]
TRAIN_IMAGES = get_dataset()["train_images"]
TRAIN_LABELS = get_dataset()["train_labels"]
VALID_IMAGES = get_dataset()["val_images"]
VALID_LABELS = get_dataset()["val_labels"]
DATA_YAML = get_dataset()["data_yaml"]
