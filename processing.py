from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BLUR_KERNEL = (5, 5)
MORPH_KERNEL = (3, 3)
MORPH_ITER = 2
AREA_MIN = 200
AREA_MAX = 50_000
MARGEM = 30
SOLIDEZ_MIN = 0.35
ASPECTO_MIN = 0.10
ASPECTO_MAX = 10.0
EXTENT_MIN = 0.15
CLOSE_KERNEL = (9, 9)
CLOSE_ITER = 1
USAR_WATERSHED = True
WATERSHED_DIST_LIMIAR = 0.3


@dataclass(slots=True)
class ProcessResult:
    count: int
    rejected: int
    mask: np.ndarray
    annotated: np.ndarray
    composite: np.ndarray


def decode_image(image_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Nao foi possivel ler a imagem enviada.")
    return image_bgr


def metricas_forma(contorno: np.ndarray) -> dict[str, float]:
    area = cv2.contourArea(contorno)
    hull_area = cv2.contourArea(cv2.convexHull(contorno))
    solidez = area / hull_area if hull_area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(contorno)
    aspecto = w / h if h > 0 else 0.0
    extent = area / (w * h) if (w * h) > 0 else 0.0
    return {
        "area": float(area),
        "solidez": float(solidez),
        "aspecto": float(aspecto),
        "extent": float(extent),
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
    }


def e_parafuso(metricas: dict[str, float]) -> tuple[bool, str]:
    if not (AREA_MIN <= metricas["area"] <= AREA_MAX):
        return False, f"area={metricas['area']:.0f}"
    if metricas["solidez"] < SOLIDEZ_MIN:
        return False, f"solidez={metricas['solidez']:.2f}"
    if not (ASPECTO_MIN <= metricas["aspecto"] <= ASPECTO_MAX):
        return False, f"aspecto={metricas['aspecto']:.2f}"
    if metricas["extent"] < EXTENT_MIN:
        return False, f"extent={metricas['extent']:.2f}"
    return True, ""


def aplicar_watershed(mask: np.ndarray, imagem_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, WATERSHED_DIST_LIMIAR * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_ws = imagem_bgr.copy()
    markers_out = cv2.watershed(img_ws, markers)
    mascara_ws = np.zeros_like(mask)
    mascara_ws[markers_out > 1] = 255
    n_regioes = int(max(markers_out.max() - 1, 0))
    return mascara_ws, markers_out, n_regioes


def add_panel_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 42), (18, 24, 38), -1)
    cv2.putText(
        labeled,
        label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (246, 247, 250),
        2,
        cv2.LINE_AA,
    )
    return labeled


def compose_result(original_rgb: np.ndarray, mask: np.ndarray, annotated_rgb: np.ndarray, count: int, rejected: int) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    original_panel = add_panel_label(original_rgb, "Original")
    mask_panel = add_panel_label(mask_rgb, "Mascara")
    annotated_panel = add_panel_label(annotated_rgb, f"Contagem: {count} | Rejeitados: {rejected}")

    height = max(original_panel.shape[0], mask_panel.shape[0], annotated_panel.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        if image.shape[0] == height:
            return image
        ratio = height / image.shape[0]
        width = int(image.shape[1] * ratio)
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    panels = [resize_to_height(panel) for panel in (original_panel, mask_panel, annotated_panel)]
    separator = np.full((height, 12, 3), 20, dtype=np.uint8)
    return np.hstack([panels[0], separator, panels[1], separator, panels[2]])


def process_image_bytes(image_bytes: bytes) -> ProcessResult:
    imagem_bgr = decode_image(image_bytes)
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, BLUR_KERNEL, 0)

    binaria = cv2.adaptiveThreshold(
        suavizada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        4,
    )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CLOSE_KERNEL)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    fechada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_close, iterations=CLOSE_ITER)
    limpa = cv2.morphologyEx(fechada, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_ITER)

    if USAR_WATERSHED:
        limpa, _, _ = aplicar_watershed(limpa, imagem_bgr)

    limpa[:MARGEM, :] = 0
    limpa[-MARGEM:, :] = 0
    limpa[:, :MARGEM] = 0
    limpa[:, -MARGEM:] = 0

    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    metricas = [metricas_forma(contorno) for contorno in contornos]

    annotated_rgb = imagem_rgb.copy()
    validos: list[tuple[np.ndarray, dict[str, float]]] = []
    rejeitados = 0

    for contorno, m in zip(contornos, metricas):
        ok, motivo = e_parafuso(m)
        if ok:
            validos.append((contorno, m))
            cv2.drawContours(annotated_rgb, [contorno], -1, (0, 200, 80), 2)
            cx = int(m["x"] + m["w"] // 2)
            cy = int(m["y"] + m["h"] // 2)
            cv2.circle(annotated_rgb, (cx, cy), 4, (255, 50, 50), -1)
            cv2.putText(
                annotated_rgb,
                str(len(validos)),
                (cx - 8, max(cy - 10, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 50, 50),
                2,
                cv2.LINE_AA,
            )
        else:
            rejeitados += 1
            cv2.drawContours(annotated_rgb, [contorno], -1, (200, 50, 50), 1)
            cv2.putText(
                annotated_rgb,
                motivo,
                (int(m["x"]), max(int(m["y"] - 4), 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (200, 50, 50),
                1,
                cv2.LINE_AA,
            )

    composite = compose_result(imagem_rgb, limpa, annotated_rgb, len(validos), rejeitados)
    return ProcessResult(
        count=len(validos),
        rejected=rejeitados,
        mask=limpa,
        annotated=annotated_rgb,
        composite=composite,
    )