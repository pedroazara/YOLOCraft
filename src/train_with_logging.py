"""
Exemplo de como usar o TrainingLogger ao treinar modelos YOLO
"""
from pathlib import Path
import time
from ultralytics import YOLO
import torch
import sys

sys.path.insert(0, str(Path.cwd().resolve().parent))

from src.training_logger import TrainingLogger
import src.config as cfg


def train_with_logging(
    model_size="n",
    epochs=100,
    batch_size=16,
    imgsz=768,
    notes="",
):
    """
    Treina modelo YOLO e registra automaticamente as informações

    Args:
        model_size: Tamanho do modelo (n, s, m, l, x)
        epochs: Número de épocas
        batch_size: Tamanho do batch
        imgsz: Tamanho da imagem
        notes: Notas sobre o treino
    """

    # Inicializar logger
    logger = TrainingLogger(log_dir="training_logs")

    print("\n" + "="*60)
    print("🚀 TREINANDO MODELO COM LOGGING")
    print("="*60)

    # Carregar modelo
    model_name = f"yolo26{model_size}"
    print(f"\n📦 Carregando modelo: {model_name}")
    model = YOLO(f"{model_name}.pt")

    # Preparar parâmetros
    training_params = {
        "data": str(cfg.DATA_YAML),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "patience": 20,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "save": True,
    }

    print(f"\n⚙️  Configuração:")
    print(f"   • Modelo: {model_name}")
    print(f"   • Épocas: {epochs}")
    print(f"   • Batch Size: {batch_size}")
    print(f"   • Tamanho da Imagem: {imgsz}")
    print(f"   • GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Registrar tempo inicial
    start_time = time.time()

    # Treinar
    print("\n⏳ Iniciando treinamento...")
    print("-"*60)

    results = model.train(**training_params)

    # Calcular tempo
    train_time = time.time() - start_time

    print("-"*60)
    print("✅ Treinamento concluído!")

    # Validar
    print("\n🔍 Validando modelo...")
    metrics = model.val(workers=0)

    # Extrair métricas por classe
    class_metrics = {}
    if hasattr(metrics, "results_dict"):
        # Processar métricas por classe
        for key, value in metrics.results_dict.items():
            if "per_class" in str(key):
                class_metrics[key] = value

    # Registrar no logger
    print("\n📝 Registrando treinamento...")
    train_id, record = logger.log_training(
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        map50=metrics.box.map50,
        map50_95=metrics.box.map,
        train_time=train_time,
        notes=notes,
        hyperparameters={
            "learning_rate": "padrão",
            "optimizer": "SGD",
            "batch_size": batch_size,
            "imgsz": imgsz,
            "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        },
    )

    # Exibir resumo
    print("\n" + "="*60)
    print("📊 RESUMO DO TREINAMENTO")
    print("="*60)
    print(f"✅ ID do Treino: {train_id}")
    print(f"📊 mAP50: {metrics.box.map50:.4f}")
    print(f"📊 mAP50-95: {metrics.box.map:.4f}")
    print(f"⏱️  Tempo: {train_time/3600:.2f} horas ({train_time:.0f} segundos)")
    print(f"📝 Notas: {notes if notes else '(nenhuma)'}")
    print("="*60)

    # Mostrar histórico
    print("\n📋 Histórico completo:")
    logger.print_summary()

    return model, logger, train_id


if __name__ == "__main__":
    # Exemplo de uso
    model, logger, train_id = train_with_logging(
        model_size="n",
        epochs=20,  # Poucos para teste
        batch_size=16,
        imgsz=768,
        notes="Teste com modelo nano para verificar logging"
    )

    # Depois você pode ver detalhes específicos:
    # logger.print_detailed(train_id)
    # logger.print_summary()
