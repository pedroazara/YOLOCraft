from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
import torch
import src.config as cfg
from src.training_logger import TrainingLogger


def train(model_size="n", epochs=100, batch_size=16, imgsz=768, notes=""):
    logger = TrainingLogger()
    model = YOLO(f"yolo26{model_size}.pt")

    start = time.time()
    model.train(
        data=str(cfg.DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=20,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    train_time = time.time() - start

    metrics = model.val(workers=0)

    train_id, _ = logger.log_training(
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        map50=metrics.box.map50,
        map50_95=metrics.box.map,
        train_time=train_time,
        notes=notes,
    )

    print(f"\nID: {train_id} | mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
    return model, logger, train_id


if __name__ == "__main__":
    train(model_size="n", epochs=100, notes="baseline")
