"""
Sincroniza o modelo de producao com o MLflow Model Registry.

Baixa a versao do modelo 'MOB_DET_YOLO' apontada pelo alias 'production'
e a copia para models/production/MOB_DET_YOLO_V1.pt (o arquivo que a API
e o Dockerfile usam). Rodar apos mover o alias no painel do MLflow.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "MOB_DET_YOLO"
ALIAS = "production"

ROOT_DIR = Path(__file__).resolve().parent.parent
PRODUCTION_PATH = ROOT_DIR / "models" / "production" / "MOB_DET_YOLO_V1.pt"


def promote():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    print(f"{MODEL_NAME} v{version.version} (alias '{ALIAS}')")
    print(f"origem: {version.source}")

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = mlflow.artifacts.download_artifacts(version.source, dst_path=tmp)
        weights = list(Path(local_dir).rglob("*.pt"))
        if not weights:
            print("erro: nenhum arquivo .pt encontrado nos artefatos da versao")
            sys.exit(1)
        if len(weights) > 1:
            print(f"erro: mais de um .pt encontrado: {[w.name for w in weights]}")
            sys.exit(1)

        PRODUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(weights[0], PRODUCTION_PATH)

    print(f"copiado para {PRODUCTION_PATH.relative_to(ROOT_DIR)}")
    print("proximo passo: commitar e enviar para o deploy (git push space deploy:main)")


if __name__ == "__main__":
    promote()
