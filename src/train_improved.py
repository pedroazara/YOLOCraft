"""
Script para retrainamento melhorado do modelo YOLO
Usa hiperparâmetros otimizados e técnicas de data augmentation
"""
from pathlib import Path
from ultralytics import YOLO
import torch

def train_improved_model(
    data_yaml,
    model_size="n",  # n, s, m, l, x
    epochs=150,
    pretrained_weights=None,
    use_best_practices=True
):
    """
    Treina modelo YOLO com melhores hiperparâmetros

    Args:
        data_yaml: Caminho para data.yaml
        model_size: Tamanho do modelo (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Número de épocas
        pretrained_weights: Caminho para pesos pré-treinados (para fine-tuning)
        use_best_practices: Se True, usa hiperparâmetros otimizados
    """

    print("="*60)
    print("🚀 TREINAR MODELO YOLO MELHORADO")
    print("="*60)

    # Selecionar modelo base
    model_name = f"yolo26{model_size}"

    if pretrained_weights:
        print(f"\n📦 Carregando pesos pré-treinados: {pretrained_weights}")
        model = YOLO(pretrained_weights)
    else:
        print(f"\n📦 Carregando modelo base: {model_name}")
        model = YOLO(f"{model_name}.pt")

    print(f"📊 Tamanho do modelo: {model_size.upper()}")
    print(f"🔢 Épocas: {epochs}")
    print(f"💾 Dataset: {data_yaml}")
    print(f"🎮 GPU disponível: {torch.cuda.is_available()}")

    # Hiperparâmetros otimizados
    if use_best_practices:
        print("\n✨ Usando hiperparâmetros otimizados...")
        training_params = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": 768,  # Tamanho mantido
            "batch": 16,
            "patience": 20,  # Early stopping

            # Data augmentation agressiva
            "hsv_h": 0.015,  # Image HSV-Hue augmentation
            "hsv_s": 0.7,    # Image HSV-Saturation augmentation
            "hsv_v": 0.4,    # Image HSV-Value augmentation
            "degrees": 10.0,  # Image rotation
            "translate": 0.1, # Image translation
            "scale": 0.5,     # Image scale
            "flipud": 0.5,    # Image flip up-down
            "fliplr": 0.5,    # Image flip left-right
            "mosaic": 1.0,    # Image mosaic
            "mixup": 0.2,     # Image mixup
            "copy_paste": 0.2,# Copy-paste augmentation

            # Otimização
            "optimizer": "SGD",  # ou "Adam"
            "lr0": 0.01,    # Learning rate inicial
            "lrf": 0.01,    # Learning rate final
            "momentum": 0.937,
            "weight_decay": 0.0005,

            # Outros
            "device": 0,    # GPU 0
            "workers": 4,
            "save": True,
            "save_period": 10,  # Salvar a cada 10 épocas
            "close_mosaic": 10, # Desabilitar mosaic 10 épocas antes do final
        }
    else:
        training_params = {
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": 768,
            "batch": 16,
        }

    print("\n⏳ Iniciando treinamento...")
    print("-"*60)

    results = model.train(**training_params)

    print("\n" + "="*60)
    print("✅ TREINAMENTO CONCLUÍDO")
    print("="*60)
    print(f"📁 Resultados salvos em: {results.save_dir}")

    # Validar modelo
    print("\n🔍 Validando modelo...")
    metrics = model.val()

    print(f"\n📊 Métricas Finais:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")

    return model, results


def compare_model_sizes(data_yaml, epochs=50):
    """
    Compara diferentes tamanhos de modelos YOLO
    """
    print("\n" + "="*60)
    print("⚖️  COMPARANDO TAMANHOS DE MODELOS")
    print("="*60)

    model_sizes = ["n", "s", "m"]  # nano, small, medium
    results_comparison = []

    for size in model_sizes:
        print(f"\n🔄 Treinando YOLO26{size}...")
        try:
            model, results = train_improved_model(
                data_yaml=data_yaml,
                model_size=size,
                epochs=epochs,
                use_best_practices=True
            )

            metrics = model.val()
            results_comparison.append({
                'size': size,
                'map50': metrics.box.map50,
                'map': metrics.box.map
            })
        except Exception as e:
            print(f"❌ Erro ao treinar YOLO26{size}: {e}")

    # Exibir comparação
    print("\n" + "="*60)
    print("📈 COMPARAÇÃO DE MODELOS")
    print("="*60)
    print(f"{'Tamanho':<10} {'mAP50':<12} {'mAP50-95':<12}")
    print("-"*40)
    for item in results_comparison:
        print(f"YOLO26{item['size']:<4} {item['map50']:<12.4f} {item['map']:<12.4f}")


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path.cwd().resolve().parent))
    import src.config as cfg

    DATA_YAML = cfg.DATA_YAML

    # Opção 1: Fine-tuning simples com melhorados parâmetros
    print("\n📌 OPÇÃO 1: Fine-tuning com hiperparâmetros otimizados\n")
    model, results = train_improved_model(
        data_yaml=DATA_YAML,
        model_size="n",  # Manter tamanho atual
        epochs=100,
        use_best_practices=True
    )

    # Descomente para testar modelos maiores também:
    # print("\n📌 OPÇÃO 2: Comparar diferentes tamanhos\n")
    # compare_model_sizes(DATA_YAML, epochs=50)
