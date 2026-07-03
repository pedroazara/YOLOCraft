"""
Wrapper de segmentação via SAM/MobileSAM, no mesmo formato usado pelo módulo
de segmentação clássica (segmentation.py), para que a interface gráfica possa
alternar entre os dois de forma intercambiável.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import SAM, YOLO

from src.segmentation.classic_segmentation import mask_to_polygon


def _color_for_class(name: str) -> tuple[int, int, int]:
    """Gera uma cor BGR estável a partir do nome da classe (mesma cor sempre para a mesma classe)."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class SamMobSegmenter:
    """Encapsula o pipeline de detecção (YOLO) + segmentação via SAM/MobileSAM."""

    def __init__(
        self,
        det_model: str | Path | YOLO,
        sam_model: str | Path | SAM,
        conf_threshold: float = 0.25,
        poly_epsilon: float = 1.5,
    ) -> None:
        # aceita caminho ou instância já carregada, pra poder reaproveitar modelos
        self.det_model = det_model if isinstance(det_model, YOLO) else YOLO(str(det_model))
        self.sam_model = sam_model if isinstance(sam_model, SAM) else SAM(str(sam_model))
        self.conf_threshold = conf_threshold
        self.poly_epsilon = poly_epsilon

    @property
    def class_names(self) -> dict[int, str]:
        return self.det_model.names

    def detect_and_segment(self, image: np.ndarray) -> dict[str, Any]:
        """Roda detecção (YOLO) + segmentação (SAM) numa imagem BGR e devolve um dict serializável."""
        height, width = image.shape[:2]
        det_result = self.det_model.predict(image, conf=self.conf_threshold, verbose=False)[0]

        detections: list[dict[str, Any]] = []
        if len(det_result.boxes) == 0:
            return {"width": width, "height": height, "method": "sam", "detections": detections}

        boxes = det_result.boxes.xyxy.cpu().numpy()
        confs = det_result.boxes.conf.cpu().numpy()
        classes = det_result.boxes.cls.cpu().numpy()

        sam_result = self.sam_model(image, bboxes=boxes.tolist(), verbose=False)
        masks = (
            sam_result[0].masks.data.cpu().numpy()
            if sam_result and sam_result[0].masks is not None
            else []
        )

        for i, box in enumerate(boxes):
            mask = masks[i] if i < len(masks) else None
            polygon = mask_to_polygon(mask, self.poly_epsilon) if mask is not None else []
            detections.append(
                {
                    "class": self.class_names[int(classes[i])],
                    "confidence": float(confs[i]),
                    "box": [float(x) for x in box],
                    "polygon": polygon,
                }
            )

        return {"width": width, "height": height, "method": "sam", "detections": detections}

    def draw_detections(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Desenha caixas, rótulos e polígonos de segmentação sobre uma cópia da imagem."""
        out = image.copy()
        overlay = image.copy()

        for det in detections:
            color = _color_for_class(det["class"])
            polygon = det.get("polygon") or []

            if len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

            x1, y1, x2, y2 = (int(v) for v in det["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f'{det["class"]} {det["confidence"]:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
