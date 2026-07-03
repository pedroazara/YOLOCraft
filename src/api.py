from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from ultralytics import YOLO, SAM

from segmentation import MobSegmenter  # segmentação clássica: otsu / hsv / grabcut

ROOT_DIR = Path(__file__).resolve().parent.parent
DET_MODEL_PATH = ROOT_DIR / "notebooks" / "3_experimentos" / "runs" / "detect" / "train" / "weights" / "best.pt"
SAM_MODEL_PATH = ROOT_DIR / "pretrained_models" / "mobile_sam.pt"

app = FastAPI(title="YOLOCraft API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
def root():
    return {"status": "YOLOCraft API running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Segmentação via SAM (seu pipeline original)
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detecção (YOLO) + segmentação via SAM."""
    contents = await file.read()
    image = _read_image(contents)
    height, width = image.shape[:2]

    r = det_model.predict(image, conf=0.25, verbose=False)[0]

    detections = []
    if len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        sam_result = sam_model(image, bboxes=boxes.tolist(), verbose=False)
        masks = sam_result[0].masks.data.cpu().numpy() if sam_result[0].masks is not None else []

        for i, box in enumerate(boxes):
            polygon = mask_to_polygon(masks[i]) if i < len(masks) else []
            detections.append({
                "class": det_model.names[int(r.boxes.cls[i])],
                "confidence": float(r.boxes.conf[i]),
                "box": [float(x) for x in box],
                "polygon": polygon,
            })

    return {"width": width, "height": height, "method": "sam", "detections": detections}


# ---------------------------------------------------------------------------
# Segmentação clássica (otsu / hsv / grabcut) — sem SAM
# ---------------------------------------------------------------------------
@app.post("/predict/classic")
async def predict_classic(
    file: UploadFile = File(...),
    method: str = Query("auto", description="otsu, hsv, grabcut ou auto"),
):
    """Detecção (YOLO) + segmentação clássica (otsu/hsv/grabcut), sem SAM."""
    contents = await file.read()
    image = _read_image(contents)
    return classic_segmenter.detect_and_segment(image, method=method)


@app.post("/predict/classic/visualize")
async def predict_classic_visualize(
    file: UploadFile = File(...),
    method: str = Query("auto", description="otsu, hsv, grabcut ou auto"),
):
    """Mesma coisa que /predict/classic, mas devolve um PNG anotado (debug visual)."""
    contents = await file.read()
    image = _read_image(contents)
    result = classic_segmenter.detect_and_segment(image, method=method)
    annotated = classic_segmenter.draw_detections(image, result["detections"])
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar imagem anotada.")
    return Response(content=buf.tobytes(), media_type="image/png")
