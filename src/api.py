import hashlib
import random
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO, SAM

from src.segmentation import MobSegmenter  # segmentação clássica: otsu / hsv / grabcut / watershed

ROOT_DIR = Path(__file__).resolve().parent.parent
DET_MODEL_PATH = ROOT_DIR / "models" / "production" / "MOB_DET_YOLO_V1.pt"
SAM_MODEL_PATH = ROOT_DIR / "pretrained_models" / "mobile_sam.pt"

# imagens de teste para o site, no mesmo layout images/labels/data.yaml usado no resto do projeto
SAMPLES_DIR = ROOT_DIR / "static" / "samples"
SAMPLES_IMAGES_DIR = SAMPLES_DIR / "images"
SAMPLES_LABELS_DIR = SAMPLES_DIR / "labels"
SAMPLES_YAML = SAMPLES_DIR / "data.yaml"
SAMPLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="YOLOCraft API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_LABELS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

# --- modelos ---
det_model = YOLO(str(DET_MODEL_PATH))
sam_model = SAM(str(SAM_MODEL_PATH))

# reaproveita o MESMO det_model já carregado acima (não recarrega o YOLO)
classic_segmenter = MobSegmenter(det_model, default_method="auto")

print("YOLO classes:", det_model.names)
print("SAM loaded from:", SAM_MODEL_PATH)


def _read_image(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem enviada.")
    return image


def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    poly = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(poly, 1.5, True)
    return poly.reshape(-1, 2).tolist()


# cache pequeno da última detecção por imagem: evita rodar o YOLO de novo quando o
# modo comparativo do site testa vários métodos de segmentação na mesma imagem
_DETECTION_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
_DETECTION_CACHE_MAX = 32


def _detect(image: np.ndarray, image_hash: str):
    cached = _DETECTION_CACHE.get(image_hash)
    if cached is not None:
        _DETECTION_CACHE.move_to_end(image_hash)
        return cached

    r = det_model.predict(image, conf=0.25, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    result = (boxes, confs, classes)

    _DETECTION_CACHE[image_hash] = result
    if len(_DETECTION_CACHE) > _DETECTION_CACHE_MAX:
        _DETECTION_CACHE.popitem(last=False)
    return result


@app.get("/")
def root():
    return {"status": "YOLOCraft API running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ping")
def ping():
    """Checagem mínima de disponibilidade: responde 'pong' se a API estiver de pé."""
    return {"ping": "pong"}


def _sample_class_names() -> dict[int, str]:
    if not SAMPLES_YAML.exists():
        return {}
    data = yaml.safe_load(SAMPLES_YAML.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def _classes_in_label(label_path: Path, names: dict[int, str]) -> list[str]:
    if not label_path.exists():
        return []
    found = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        found.add(names.get(int(parts[0]), parts[0]))
    return sorted(found)


def _list_samples() -> list[dict]:
    names = _sample_class_names()
    files = sorted(
        p for p in SAMPLES_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS
    )
    return [
        {
            "name": f.name,
            "url": f"/static/samples/images/{f.name}",
            "classes": _classes_in_label(SAMPLES_LABELS_DIR / f"{f.stem}.txt", names),
        }
        for f in files
    ]


@app.get("/samples/classes")
def list_sample_classes():
    """Lista os nomes de mob presentes nas imagens de teste (para autocomplete no site)."""
    classes = sorted({c for sample in _list_samples() for c in sample["classes"]})
    return {"classes": classes}


@app.get("/samples")
def list_samples(
    mob: str | None = Query(None, description="filtra por nome do mob (ex: creeper); vazio = qualquer um"),
    count: int = Query(4, ge=1, le=50, description="quantidade de imagens aleatórias a devolver"),
):
    """Devolve imagens de teste escolhidas aleatoriamente, opcionalmente filtradas por mob."""
    samples = _list_samples()
    if mob:
        mob = mob.strip().lower()
        samples = [s for s in samples if any(mob in c.lower() for c in s["classes"])]
    random.shuffle(samples)
    return {"samples": samples[:count]}


# ---------------------------------------------------------------------------
# Segmentação via SAM (seu pipeline original)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detecção (YOLO) + segmentação via SAM."""
    contents = await file.read()
    image = _read_image(contents)
    height, width = image.shape[:2]

    image_hash = hashlib.sha256(contents).hexdigest()
    boxes, confs, classes = _detect(image, image_hash)

    detections = []
    if len(boxes):
        sam_result = sam_model(image, bboxes=boxes.tolist(), verbose=False)
        masks = sam_result[0].masks.data.cpu().numpy() if sam_result[0].masks is not None else []

        for i, box in enumerate(boxes):
            polygon = mask_to_polygon(masks[i]) if i < len(masks) else []
            detections.append({
                "class": det_model.names[int(classes[i])],
                "confidence": float(confs[i]),
                "box": [float(x) for x in box],
                "polygon": polygon,
            })

    return {"width": width, "height": height, "method": "sam", "detections": detections}


# ---------------------------------------------------------------------------
# Segmentação clássica (otsu / hsv / grabcut / watershed) — sem SAM
# ---------------------------------------------------------------------------
def _classic_params(
    method: Literal["otsu", "hsv", "grabcut", "watershed", "auto"] = Query("auto"),
    margin_ratio: float = Query(0.25, description="contexto ao redor da box (fração da largura/altura), usado por todos os métodos"),
    grabcut_iterations: int = Query(5, ge=1, le=20, description="iterações do GrabCut (só método grabcut/auto)"),
    hsv_threshold: float = Query(2.2, gt=0, description="sensibilidade do método hsv — menor = mais sensível (só método hsv/auto)"),
    watershed_fg_ratio: float = Query(0.5, gt=0, lt=1, description="fração da distance transform tratada como primeiro plano (só método watershed)"),
    poly_epsilon: float = Query(1.5, gt=0, description="simplificação do polígono final, usado por todos os métodos"),
):
    return {
        "method": method,
        "margin_ratio": margin_ratio,
        "grabcut_iterations": grabcut_iterations,
        "hsv_threshold": hsv_threshold,
        "watershed_fg_ratio": watershed_fg_ratio,
        "poly_epsilon": poly_epsilon,
    }


@app.post("/predict/classic")
async def predict_classic(file: UploadFile = File(...), params: dict = Depends(_classic_params)):
    """Detecção (YOLO) + segmentação clássica (otsu/hsv/grabcut/watershed), sem SAM."""
    contents = await file.read()
    image = _read_image(contents)
    image_hash = hashlib.sha256(contents).hexdigest()
    precomputed = _detect(image, image_hash)
    return classic_segmenter.detect_and_segment(image, precomputed_detection=precomputed, **params)


@app.post("/predict/classic/visualize")
async def predict_classic_visualize(file: UploadFile = File(...), params: dict = Depends(_classic_params)):
    """Mesma coisa que /predict/classic, mas devolve um PNG anotado (debug visual)."""
    contents = await file.read()
    image = _read_image(contents)
    image_hash = hashlib.sha256(contents).hexdigest()
    precomputed = _detect(image, image_hash)
    result = classic_segmenter.detect_and_segment(image, precomputed_detection=precomputed, **params)
    annotated = classic_segmenter.draw_detections(image, result["detections"])
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar imagem anotada.")
    return Response(content=buf.tobytes(), media_type="image/png")
