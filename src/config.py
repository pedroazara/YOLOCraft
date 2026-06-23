from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data" / "minecraft_mobs" / "minecraft_mobs_yolo"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "val" / "images"
VALID_LABELS = DATASET_DIR / "val" / "labels"


DATA_YAML = DATASET_DIR / "data.yaml"