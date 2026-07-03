from pathlib import Path
from ultralytics import YOLO
import cv2


def test_thresholds(model_path, image_path, thresholds=None, save_dir="threshold_tests"):
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    model = YOLO(model_path)
    results_summary = []

    for conf in thresholds:
        result = model.predict(source=str(image_path), conf=conf, save=False, verbose=False)[0]
        num_detections = len(result.boxes)

        class_counts = {}
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            class_counts[name] = class_counts.get(name, 0) + 1

        results_summary.append({"conf": conf, "total": num_detections, "classes": class_counts})

        annotated = result.plot()
        cv2.imwrite(str(save_path / f"conf_{conf:.2f}.jpg"), annotated)

        details = " | ".join(f"{k}:{v}" for k, v in class_counts.items()) or "no detections"
        print(f"conf={conf:.2f}  detections={num_detections}  {details}")

    return results_summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    test_thresholds(
        model_path=str(root / "notebooks/runs/detect/train-4/weights/best.pt"),
        image_path=str(root / "notebooks/testes/spiders.jpg"),
    )
