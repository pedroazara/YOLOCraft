import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import torch


class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.json_file = self.log_dir / "training_history.json"
        self.csv_file = self.log_dir / "training_history.csv"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        if self.json_file.exists():
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_csv(self):
        if not self.history:
            return
        keys = self.history[0].keys()
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def log_training(
        self,
        model_size: str,
        epochs: int,
        batch_size: int,
        imgsz: int,
        map50: float,
        map50_95: float,
        train_time: float,
        notes: str = "",
        hyperparameters: Dict[str, Any] = None,
        class_metrics: Dict[str, Dict] = None,
    ):
        timestamp = datetime.now()
        train_id = timestamp.strftime("%Y%m%d_%H%M%S")

        record = {
            "id": train_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%d/%m/%Y"),
            "time": timestamp.strftime("%H:%M:%S"),
            "model": f"yolo26{model_size}",
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "train_time_hours": round(train_time / 3600, 2),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "notes": notes,
            "hyperparameters": hyperparameters or {},
            "class_metrics": class_metrics or {},
        }

        self.history.append(record)
        self._save_history()
        self._save_csv()
        return train_id, record

    def get_best_model(self, metric: str = "map50") -> Dict[str, Any]:
        if not self.history:
            return None
        return max(self.history, key=lambda x: x.get(metric, 0))

    def get_by_model(self, model: str) -> List[Dict[str, Any]]:
        return [r for r in self.history if r["model"] == model]

    def summary(self):
        if not self.history:
            print("No training records found.")
            return

        print(f"\n{'ID':<18} {'Model':<12} {'Epochs':<8} {'mAP50':<10} {'mAP50-95':<10} {'Time(h)':<8} {'Date'}")
        print("-" * 80)

        for r in sorted(self.history, key=lambda x: x["timestamp"], reverse=True):
            print(f"{r['id']:<18} {r['model']:<12} {r['epochs']:<8} {r['map50']:<10.4f} {r['map50_95']:<10.4f} {r['train_time_hours']:<8.2f} {r['date']}")

        best = self.get_best_model("map50")
        print(f"\nBest: {best['model']} (ID: {best['id']}) — mAP50: {best['map50']:.4f}")
