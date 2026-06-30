"""
Script para testar diferentes confidence thresholds no modelo YOLO
"""
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def test_thresholds(model_path, image_path, thresholds=None, save_dir="threshold_tests"):
    """
    Testa diferentes confidence thresholds e visualiza os resultados

    Args:
        model_path: Caminho para o modelo .pt
        image_path: Caminho para a imagem de teste
        thresholds: Lista de thresholds para testar
        save_dir: Diretório para salvar as imagens
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Criar diretório de saída
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Carregar modelo
    model = YOLO(model_path)
    image_path = Path(image_path)

    print(f"🔍 Testando modelo: {model_path}")
    print(f"📷 Imagem: {image_path}")
    print(f"📊 Thresholds: {thresholds}\n")

    results_summary = []

    # Testar cada threshold
    for conf in thresholds:
        print(f"⏳ Testando conf={conf}...", end=" ")

        results = model.predict(
            source=str(image_path),
            conf=conf,
            save=False,
            verbose=False
        )

        # Contar detecções
        detections = results[0]
        num_detections = len(detections.boxes)

        # Agrupar por classe
        class_counts = {}
        if num_detections > 0:
            for box in detections.boxes:
                class_id = int(box.cls[0])
                class_name = detections.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        results_summary.append({
            'conf': conf,
            'total': num_detections,
            'classes': class_counts
        })

        # Exibir resultado
        if num_detections > 0:
            details = " | ".join([f"{k}:{v}" for k, v in class_counts.items()])
            print(f"✅ {num_detections} detecções ({details})")
        else:
            print("❌ Nenhuma detecção")

        # Salvar imagem
        annotated_image = detections.plot()
        output_file = save_path / f"conf_{conf:.2f}.jpg"
        cv2.imwrite(str(output_file), annotated_image)

    # Exibir resumo
    print("\n" + "="*60)
    print("📈 RESUMO DOS TESTES")
    print("="*60)
    print(f"{'Conf':<8} {'Total':<8} {'Detalhes':<40}")
    print("-"*60)

    for item in results_summary:
        conf = item['conf']
        total = item['total']
        details = " | ".join([f"{k}:{v}" for k, v in item['classes'].items()])
        if total == 0:
            details = "(nenhuma)"
        print(f"{conf:<8.2f} {total:<8} {details:<40}")

    # Achar melhor threshold
    best_result = max(results_summary, key=lambda x: x['total'])
    if best_result['total'] > 0:
        print("\n" + "="*60)
        print(f"🏆 MELHOR THRESHOLD: conf={best_result['conf']}")
        print(f"   Detecções: {best_result['total']}")
        print("="*60)
    else:
        print("\n⚠️  Nenhum threshold detectou objetos")

    print(f"\n💾 Imagens salvas em: {save_path.absolute()}")
    return results_summary


if __name__ == "__main__":
    # Configuração
    root_dir = Path(__file__).resolve().parent.parent
    MODEL_PATH = root_dir / "notebooks/runs/detect/train-4/weights/best.pt"
    IMAGE_PATH = root_dir / "notebooks/testes/spiders.jpg"

    # Executar testes
    results = test_thresholds(str(MODEL_PATH), str(IMAGE_PATH))
