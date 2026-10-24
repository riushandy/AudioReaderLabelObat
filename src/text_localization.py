"""Tahap 2 - Lokalisasi teks (image segmentation berbasis kontur).

Ide: huruf menghasilkan tepi (edge) yang rapat dan berulang secara horizontal.
1. Morphological gradient -> menonjolkan tepi huruf (tidak peduli teks gelap
   di latar terang atau teks terang di latar gelap).
2. Otsu threshold -> peta tepi biner.
3. Closing dengan kernel horizontal -> huruf-huruf dalam satu baris menyatu
   menjadi satu blob.
4. findContours + filter geometri (ukuran, rasio aspek, kepadatan tepi)
   -> kandidat ROI baris teks.
5. Gabung kotak yang tumpang tindih, urutkan sesuai urutan baca.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import LocalizationConfig


@dataclass
class TextBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self):
        return self.x, self.y, self.w, self.h


def edge_map(gray: np.ndarray, cfg: LocalizationConfig) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.grad_kernel)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw


def remove_long_lines(edges: np.ndarray, min_len_ratio: float = 0.18) -> np.ndarray:
    """Hapus garis lurus panjang (bingkai label, tepi kemasan, garis tabel)
    dengan Probabilistic Hough Transform agar tidak menyatu dengan blok teks."""
    H, W = edges.shape[:2]
    min_len = int(min(W, H) * min_len_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(60, min_len // 2),
                            minLineLength=min_len, maxLineGap=4)
    out = edges.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 0, 5)
    return out


def text_mask(gray: np.ndarray, cfg: LocalizationConfig) -> tuple[np.ndarray, np.ndarray]:
    edges = edge_map(gray, cfg)
    if cfg.remove_lines:
        edges = remove_long_lines(edges)
    kw = max(9, int(gray.shape[1] * cfg.close_kernel_ratio))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, cfg.close_kernel_h))
    mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_k)
    # buka sedikit untuk memutus sambungan tipis antar baris
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return edges, mask


def _merge_boxes(boxes: list[TextBox], iou_thr: float = 0.25) -> list[TextBox]:
    """Gabungkan kotak yang saling tumpang tindih / bersinggungan di baris sama."""
    changed = True
    boxes = boxes[:]
    while changed:
        changed = False
        out: list[TextBox] = []
        while boxes:
            a = boxes.pop(0)
            i = 0
            while i < len(boxes):
                b = boxes[i]
                ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
                iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
                inter = ix * iy
                small = min(a.area, b.area)
                same_line = iy > 0.6 * min(a.h, b.h) and (
                    max(a.x, b.x) - min(a.x + a.w, b.x + b.w) < 0.8 * max(a.h, b.h))
                if (small and inter / small > iou_thr) or same_line:
                    x1, y1 = min(a.x, b.x), min(a.y, b.y)
                    x2, y2 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                    a = TextBox(x1, y1, x2 - x1, y2 - y1)
                    boxes.pop(i)
                    changed = True
                else:
                    i += 1
            out.append(a)
        boxes = out
    return boxes


def reading_order(boxes: list[TextBox]) -> list[TextBox]:
    """Urutkan atas->bawah, lalu kiri->kanan untuk kotak di baris yang sama."""
    boxes = sorted(boxes, key=lambda b: b.y)
    lines: list[list[TextBox]] = []
    for b in boxes:
        if lines:
            ref = lines[-1][0]
            if abs((b.y + b.h / 2) - (ref.y + ref.h / 2)) < 0.5 * min(b.h, ref.h):
                lines[-1].append(b)
                continue
        lines.append([b])
    return [b for line in lines for b in sorted(line, key=lambda t: t.x)]


def localize_text(gray: np.ndarray, cfg: LocalizationConfig) -> dict:
    """Return dict: boxes (list[TextBox]), edges, mask."""
    H, W = gray.shape[:2]
    edges, mask = text_mask(gray, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < cfg.min_box_h or w < cfg.min_box_w or h > cfg.max_box_h_ratio * H:
            continue
        if w / float(h) < cfg.min_aspect:
            continue
        fill = cv2.countNonZero(edges[y:y + h, x:x + w]) / float(w * h)
        if fill < cfg.min_fill * 0.5 or fill > 0.95:
            continue
        # buang kotak yang menempel di seluruh tepi (bingkai / tepi kemasan)
        if w > 0.97 * W:
            continue
        boxes.append(TextBox(x, y, w, h))
    boxes = _merge_boxes(boxes)
    boxes = [b for b in boxes if b.h <= cfg.max_box_h_ratio * H]
    return {"boxes": reading_order(boxes), "edges": edges, "mask": mask}


def crop(img: np.ndarray, box: TextBox, pad: int) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1 = max(0, box.x - pad), max(0, box.y - pad)
    x2, y2 = min(W, box.x + box.w + pad), min(H, box.y + box.h + pad)
    return img[y1:y2, x1:x2]
