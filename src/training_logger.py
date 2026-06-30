"""
Sistema de logging para rastrear treinos de modelos YOLO
Mantém registro de todos os treinos realizados
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import torch


class TrainingLogger:
    """Rastreia e registra informações de treinos YOLO"""

    def __init__(self, log_dir="training_logs"):
        """
        Inicializa o logger de treinos

        Args:
            log_dir: Diretório para salvar os registros
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.json_file = self.log_dir / "training_history.json"
        self.csv_file = self.log_dir / "training_history.csv"

        # Carregar histórico existente
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Carrega histórico de treinos anterior"""
        if self.json_file.exists():
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Salva histórico em JSON"""
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _save_csv(self):
        """Salva histórico em CSV para visualização rápida"""
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
        """
        Registra informações de um treino

        Args:
            model_size: Tamanho do modelo (n, s, m, l, x)
            epochs: Número de épocas treinadas
            batch_size: Tamanho do batch
            imgsz: Tamanho da imagem
            map50: mAP50 alcançado
            map50_95: mAP50-95 alcançado
            train_time: Tempo de treino em segundos
            notes: Notas adicionais sobre o treino
            hyperparameters: Dicionário com hiperparâmetros usados
            class_metrics: Métricas por classe
        """

        timestamp = datetime.now()
        train_id = timestamp.strftime("%Y%m%d_%H%M%S")

        record = {
            "id": train_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%d/%m/%Y"),
            "time": timestamp.strftime("%H:%M:%S"),
            "model_size": model_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "train_time_seconds": round(train_time, 2),
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
        """
        Retorna o melhor treino baseado em uma métrica

        Args:
            metric: Métrica para comparar (map50 ou map50_95)

        Returns:
            Dicionário com o melhor treino
        """
        if not self.history:
            return None

        return max(self.history, key=lambda x: x.get(metric, 0))

    def get_models_by_size(self, size: str) -> List[Dict[str, Any]]:
        """Retorna todos os treinos de um tamanho específico"""
        return [m for m in self.history if m["model_size"] == size]

    def print_summary(self):
        """Imprime resumo de todos os treinos"""
        if not self.history:
            print("❌ Nenhum treino registrado ainda")
            return

        print("\n" + "="*80)
        print("📊 HISTÓRICO DE TREINOS")
        print("="*80)
        print(
            f"{'ID':<18} {'Modelo':<10} {'Épocas':<8} {'mAP50':<10} "
            f"{'mAP50-95':<10} {'Tempo (h)':<10} {'Data/Hora':<20}"
        )
        print("-"*80)

        for record in sorted(self.history, key=lambda x: x["timestamp"], reverse=True):
            model = f"YOLO26{record['model_size']}"
            print(
                f"{record['id']:<18} {model:<10} {record['epochs']:<8} "
                f"{record['map50']:<10.4f} {record['map50_95']:<10.4f} "
                f"{record['train_time_hours']:<10.2f} {record['date']} {record['time']:<8}"
            )

        # Melhor modelo
        best = self.get_best_model("map50")
        print("\n" + "="*80)
        print(f"🏆 MELHOR MODELO: YOLO26{best['model_size']} (ID: {best['id']})")
        print(f"   mAP50:    {best['map50']:.4f}")
        print(f"   mAP50-95: {best['map50_95']:.4f}")
        print("="*80 + "\n")

    def print_detailed(self, train_id: str):
        """Imprime detalhes completos de um treino"""
        record = next((m for m in self.history if m["id"] == train_id), None)

        if not record:
            print(f"❌ Treino '{train_id}' não encontrado")
            return

        print("\n" + "="*60)
        print(f"📋 DETALHES DO TREINO: {train_id}")
        print("="*60)

        print(f"\n📅 Data/Hora: {record['date']} {record['time']}")
        print(f"🤖 Modelo: YOLO26{record['model_size']}")
        print(f"\n⚙️  Configuração:")
        print(f"   • Épocas: {record['epochs']}")
        print(f"   • Batch Size: {record['batch_size']}")
        print(f"   • Tamanho da Imagem: {record['imgsz']}")
        print(f"   • GPU: {record['gpu']}")

        print(f"\n📊 Resultados:")
        print(f"   • mAP50: {record['map50']:.4f}")
        print(f"   • mAP50-95: {record['map50_95']:.4f}")

        print(f"\n⏱️  Tempo de Treino:")
        print(f"   • {record['train_time_seconds']:.2f} segundos")
        print(f"   • {record['train_time_hours']:.2f} horas")

        if record["notes"]:
            print(f"\n📝 Notas: {record['notes']}")

        if record["hyperparameters"]:
            print(f"\n🔧 Hiperparâmetros:")
            for key, value in record["hyperparameters"].items():
                print(f"   • {key}: {value}")

        if record["class_metrics"]:
            print(f"\n📈 Métricas por Classe:")
            for class_name, metrics in record["class_metrics"].items():
                print(f"   {class_name}:")
                for key, value in metrics.items():
                    print(f"      • {key}: {value:.4f}")

        print("="*60 + "\n")

    def compare_models(self, size1: str, size2: str = None):
        """Compara treinos de diferentes tamanhos de modelos"""
        print("\n" + "="*80)
        print("⚖️  COMPARAÇÃO DE MODELOS")
        print("="*80)

        models = [self.get_best_model("map50")]

        for size in [size1, size2] if size2 else [size1]:
            trainings = self.get_models_by_size(size)
            if trainings:
                best = max(trainings, key=lambda x: x["map50"])
                models.append(best)

        print(f"{'Modelo':<15} {'mAP50':<12} {'mAP50-95':<12} {'Tempo (h)':<12} {'ID':<18}")
        print("-"*80)

        for model in models:
            print(
                f"YOLO26{model['model_size']:<11} {model['map50']:<12.4f} "
                f"{model['map50_95']:<12.4f} {model['train_time_hours']:<12.2f} {model['id']:<18}"
            )

        print("="*80 + "\n")

    def export_best_model_info(self) -> Dict[str, Any]:
        """Retorna informações do melhor modelo para usar em produção"""
        best = self.get_best_model("map50")
        if not best:
            return None

        return {
            "model_id": best["id"],
            "model_size": best["model_size"],
            "map50": best["map50"],
            "map50_95": best["map50_95"],
            "epochs": best["epochs"],
            "date": best["date"],
            "time": best["time"],
        }


# Exemplo de uso
if __name__ == "__main__":
    logger = TrainingLogger()

    # Simular um treino
    logger.log_training(
        model_size="n",
        epochs=100,
        batch_size=16,
        imgsz=768,
        map50=0.92,
        map50_95=0.77,
        train_time=3600,  # 1 hora
        notes="Primeiro treino baseline",
        hyperparameters={
            "learning_rate": 0.01,
            "optimizer": "SGD",
            "mosaic": 1.0,
        },
        class_metrics={
            "spider": {"precision": 0.95, "recall": 0.90, "map50": 0.96},
            "creeper": {"precision": 0.91, "recall": 0.81, "map50": 0.91},
        },
    )

    # Mostrar resumo
    logger.print_summary()
