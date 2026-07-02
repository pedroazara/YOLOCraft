from fastapi import FastAPI

from pathlib import Path
from ultralytics import YOLO, SAM

ROOT_DIR = Path(__file__).resolve().parent.parent
DET_MODEL_PATH = ROOT_DIR / "notebooks" / "3_experimentos" / "runs" / "detect" / "train-2" / "weights" / "best.pt"
SAM_MODEL_PATH = ROOT_DIR / "pretrained_models" / "mobile_sam.pt"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

det_model = YOLO(str(DET_MODEL_PATH))
sam_model = SAM(str(SAM_MODEL_PATH))

print("YOLO classes:", det_model.names)
print("SAM loaded from:", SAM_MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi import UploadFile, File
import cv2
import numpy as np

def mask_to_polygon(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    poly = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(poly, 1.5, True)
    return poly.reshape(-1, 2).tolist()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "could not decode image"}

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

    return {"width": width, "height": height, "detections": detections}

@app.get("/")
def root():
    return {"status": "YOLOCraft API running", "docs": "/docs"}