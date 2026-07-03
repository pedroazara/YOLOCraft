"""
Segmentação clássica (sem deep learning) para mobs do Minecraft.

As texturas de mobs no Minecraft são "pixel art" (poucas cores, bordas duras,
pouco ou nenhum anti-aliasing), então o pipeline aqui evita blur agressivo,
prefere filtros que preservam bordas e usa a bounding box do YOLO para
restringir o problema a uma região pequena — isso torna otsu/hsv/grabcut bem
mais confiáveis do que aplicá-los na imagem inteira.

Métodos disponíveis:
- "otsu":    limiarização de Otsu no canal de saturação (HSV), dentro da ROI.
- "hsv":     diferença de cor em HSV entre a região da box (mob) e o resto da ROI (fundo).
- "grabcut": GrabCut do OpenCV, inicializado com a própria bbox do YOLO.
- "auto":    tenta grabcut e cai para hsv/otsu se o resultado parecer ruim.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RoiCrop:
    roi: np.ndarray                          # recorte BGR (com margem) da imagem original
    x0: int                                  # offset x do recorte na imagem original
    y0: int                                  # offset y do recorte na imagem original
    box_local: tuple[int, int, int, int]     # box do mob em coords locais da ROI (x1,y1,x2,y2)


def crop_with_margin(image: np.ndarray, box: list[float], margin_ratio: float = 0.25) -> RoiCrop:
    """Recorta a região ao redor da box com uma margem, dando contexto de fundo ao grabcut/hsv."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_ratio, bh * margin_ratio

    cx1 = max(0, int(x1 - mx))
    cy1 = max(0, int(y1 - my))
    cx2 = min(w, int(x2 + mx))
    cy2 = min(h, int(y2 + my))

    roi = image[cy1:cy2, cx1:cx2]
    box_local = (int(x1 - cx1), int(y1 - cy1), int(x2 - cx1), int(y2 - cy1))
    return RoiCrop(roi=roi, x0=cx1, y0=cy1, box_local=box_local)


def _upscale_for_small_roi(roi: np.ndarray, min_side: int = 120) -> tuple[np.ndarray, float]:
    """Mobs distantes geram ROIs minúsculas (às vezes <20px); threshold, morfologia e
    contornos ficam bem mais estáveis se ampliarmos antes de segmentar."""
    h, w = roi.shape[:2]
    scale = max(1.0, min_side / max(h, w))
    if scale > 1.0:
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return roi, scale


def _largest_component_near_center(mask: np.ndarray) -> np.ndarray:
    """Mantém só o componente conectado que cobre o centro da máscara (assume-se que o
    centro da box sempre cai sobre o mob) — remove ruído de fundo que sobrou do threshold."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    center_label = labels[cy, cx]
    if center_label == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        center_label = 1 + int(np.argmax(areas))
    return np.where(labels == center_label, 255, 0).astype(np.uint8)


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Limpeza morfológica comum a todos os métodos: fecha buracos pequenos, remove
    ruído isolado, mantém só o blob principal e preenche furos internos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_component_near_center(mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def segment_otsu(roi: np.ndarray, box_local: tuple[int, int, int, int]) -> np.ndarray:
    """Otsu sobre o canal de saturação (HSV) — mobs costumam ter cores mais saturadas
    que céu/grama/pedra ao fundo, então esse canal separa melhor do que tons de cinza puro."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat = cv2.bilateralFilter(sat, d=5, sigmaColor=40, sigmaSpace=40)
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # decide qual dos dois lados do threshold é o mob: assume-se que o centro da box é mob
    x1, y1, x2, y2 = box_local
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    cy = min(max(cy, 0), mask.shape[0] - 1)
    cx = min(max(cx, 0), mask.shape[1] - 1)
    if mask[cy, cx] == 0:
        mask = cv2.bitwise_not(mask)

    return _refine_mask(mask)


def segment_hsv(roi: np.ndarray, box_local: tuple[int, int, int, int]) -> np.ndarray:
    """Modela a cor do fundo a partir da região da ROI fora da box e marca como mob todo
    pixel cuja cor em HSV se distancia o suficiente desse modelo (distância tipo z-score,
    com correção para o matiz ser circular)."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = roi.shape[:2]
    x1, y1, x2, y2 = box_local
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bg_mask = np.ones((h, w), dtype=bool)
    bg_mask[y1:y2, x1:x2] = False
    bg_pixels = hsv[bg_mask] if bg_mask.sum() >= 10 else hsv.reshape(-1, 3)

    bg_mean = bg_pixels.mean(axis=0)
    bg_std = bg_pixels.std(axis=0) + 1e-3

    diff = (hsv - bg_mean) / bg_std
    hue_diff = np.abs(diff[:, :, 0])
    diff[:, :, 0] = np.minimum(hue_diff, np.abs(180 / bg_std[0] - hue_diff))
    dist = np.sqrt((diff ** 2).sum(axis=2))

    mask = np.where(dist > 2.2, 255, 0).astype(np.uint8)
    return _refine_mask(mask)


def segment_grabcut(roi: np.ndarray, box_local: tuple[int, int, int, int], iterations: int = 5) -> np.ndarray:
    """GrabCut inicializado com a bbox do YOLO como retângulo provável de primeiro plano."""
    h, w = roi.shape[:2]
    x1, y1, x2, y2 = box_local
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # ROI degenerada (muito pequena/uniforme) — cai para a própria box como máscara
        fallback = np.zeros((h, w), np.uint8)
        fallback[y1:y2, x1:x2] = 255
        return fallback

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _refine_mask(binary)


def segment_roi(roi: np.ndarray, box_local: tuple[int, int, int, int], method: str = "grabcut") -> np.ndarray:
    """Ponto único de entrada para segmentar uma ROI já recortada, escolhendo o método."""
    roi_up, scale = _upscale_for_small_roi(roi)
    box_up = tuple(int(v * scale) for v in box_local)

    if method == "otsu":
        mask = segment_otsu(roi_up, box_up)
    elif method == "hsv":
        mask = segment_hsv(roi_up, box_up)
    elif method == "grabcut":
        mask = segment_grabcut(roi_up, box_up)
    elif method == "auto":
        mask = segment_grabcut(roi_up, box_up)
        box_area = max(1, (box_up[2] - box_up[0]) * (box_up[3] - box_up[1]))
        mask_area = int((mask > 0).sum())
        roi_area = roi_up.shape[0] * roi_up.shape[1]
        # grabcut "falhou" se pegou quase nada ou quase tudo da ROI -> tenta alternativas
        if mask_area < 0.10 * box_area or mask_area > 0.98 * roi_area:
            alt = segment_hsv(roi_up, box_up)
            mask = alt if (alt > 0).sum() >= 0.10 * box_area else segment_otsu(roi_up, box_up)
    else:
        raise ValueError(f"Método de segmentação desconhecido: {method!r}")

    if scale > 1.0:
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.5) -> list[list[int]]:
    """Converte uma máscara binária (H, W) no maior polígono de contorno externo."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    largest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return approx.reshape(-1, 2).tolist()
