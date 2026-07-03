"""
Módulo de segmentação de mobs do Minecraft.

Usa YOLO para detecção e métodos clássicos de visão computacional
(Otsu, HSV, GrabCut) para segmentação, sem depender de um segundo modelo de
deep learning. A bbox do YOLO restringe cada método a uma região pequena da
imagem, o que os torna bem mais confiáveis.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.classic_segmentation import crop_with_margin, mask_to_polygon, segment_roi


def _color_for_class(name: str) -> tuple[int, int, int]:
    """Gera uma cor BGR estável a partir do nome da classe (mesma cor sempre para a mesma classe)."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    b = int(h[0:2], 16)
    g = int(h[2:4], 16)
    r = int(h[4:6], 16)
    return (b, g, r)


class MobSegmenter:
    """Encapsula o pipeline de detecção (YOLO) + segmentação clássica (Otsu/HSV/GrabCut/Watershed)."""

    VALID_METHODS = ("otsu", "hsv", "grabcut", "watershed", "auto")

    def __init__(
        self,
        det_model: str | Path | YOLO,
        conf_threshold: float = 0.25,
        poly_epsilon: float = 1.5,
        default_method: str = "auto",
    ) -> None:
        # aceita um caminho (carrega um YOLO novo) ou uma instância já carregada
        # (reaproveita, útil quando o mesmo YOLO já é usado em outro lugar da API)
        self.det_model = det_model if isinstance(det_model, YOLO) else YOLO(str(det_model))
        self.conf_threshold = conf_threshold
        self.poly_epsilon = poly_epsilon
        if default_method not in self.VALID_METHODS:
            raise ValueError(f"Método inválido: {default_method!r}. Use um de {self.VALID_METHODS}.")
        self.default_method = default_method

    @property
    def class_names(self) -> dict[int, str]:
        return self.det_model.names

    def detect_and_segment(
        self,
        image: np.ndarray,
        method: str | None = None,
        margin_ratio: float = 0.25,
        grabcut_iterations: int = 5,
        hsv_threshold: float = 2.2,
        watershed_fg_ratio: float = 0.5,
        poly_epsilon: float | None = None,
    ) -> dict[str, Any]:
        """Roda detecção (YOLO) + segmentação clássica numa imagem BGR e devolve um dict serializável.

        Parâmetros ajustáveis (o frontend pode enviar qualquer um deles):
        - margin_ratio: contexto de fundo ao redor da box, usado por todos os métodos.
        - grabcut_iterations: só afeta o método "grabcut" (e "auto", que pode cair nele).
        - hsv_threshold: só afeta o método "hsv" (e "auto", como fallback).
        - watershed_fg_ratio: só afeta o método "watershed".
        - poly_epsilon: simplificação do polígono final, comum a todos os métodos.
        """
        method = method or self.default_method
        if method not in self.VALID_METHODS:
            raise ValueError(f"Método inválido: {method!r}. Use um de {self.VALID_METHODS}.")
        epsilon = self.poly_epsilon if poly_epsilon is None else poly_epsilon

        height, width = image.shape[:2]
        det_result = self.det_model.predict(image, conf=self.conf_threshold, verbose=False)[0]

        detections: list[dict[str, Any]] = []
        if len(det_result.boxes) == 0:
            return {"width": width, "height": height, "method": method, "detections": detections}

        boxes = det_result.boxes.xyxy.cpu().numpy()
        confs = det_result.boxes.conf.cpu().numpy()
        classes = det_result.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            crop = crop_with_margin(image, box.tolist(), margin_ratio=margin_ratio)
            roi_mask = segment_roi(
                crop.roi,
                crop.box_local,
                method=method,
                grabcut_iterations=grabcut_iterations,
                hsv_threshold=hsv_threshold,
                watershed_fg_ratio=watershed_fg_ratio,
            )
            polygon_local = mask_to_polygon(roi_mask, epsilon)
            # converte o polígono de coordenadas locais da ROI para coordenadas da imagem original
            polygon = [[px + crop.x0, py + crop.y0] for px, py in polygon_local]

            detections.append(
                {
                    "class": self.class_names[int(classes[i])],
                    "confidence": float(confs[i]),
                    "box": [float(x) for x in box],
                    "polygon": polygon,
                }
            )

        return {"width": width, "height": height, "method": method, "detections": detections}

    def draw_detections(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        """Desenha caixas, rótulos e polígonos de segmentação sobre uma cópia da imagem (debug/visualização)."""
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
