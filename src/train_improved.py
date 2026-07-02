from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
import torch
import src.config as cfg


def train(model_size="n", epochs=150, pretrained_weights=None):
    if pretrained_weights:
        model = YOLO(pretrained_weights)
    else:
        model = YOLO(f"yolo26{model_size}.pt")

    batch = {
        "n": 16,
        "s": 16,
        "m": 8,
        "l": 4,
        "x": 4,
    }.get(model_size, 16)

    results = model.train(
        data=str(cfg.DATA_YAML),
        epochs=epochs,
        imgsz=768,
        batch=batch,
        patience=20,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.2,
        close_mosaic=10,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    return model, results


if __name__ == "__main__":
    train(model_size="s")
