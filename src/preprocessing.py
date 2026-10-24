"""Tahap 1 - Pra-pemrosesan citra dengan OpenCV.

Urutan: resize -> grayscale -> CLAHE (perataan kontras) -> reduksi noise
-> deskew -> adaptive thresholding. Setiap tahap disimpan agar bisa
divisualisasikan (debug / laporan).
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import PreprocessConfig


def resize_to_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = width / float(w)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=interp), scale


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize_contrast(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """CLAHE: histogram equalization adaptif per-tile, menjaga detail pada
    kemasan yang pencahayaannya tidak merata (pantulan plastik / bayangan)."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def reduce_noise(gray: np.ndarray, method: str) -> np.ndarray:
    if method == "bilateral":      # halus tetapi menjaga tepi huruf
        return cv2.bilateralFilter(gray, 7, 50, 50)
    if method == "nlmeans":        # paling bersih, paling lambat
        return cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    if method == "median":         # efektif untuk noise salt-and-pepper
        return cv2.medianBlur(gray, 3)
    return gray


def adaptive_binarize(gray: np.ndarray, block: int, c: int) -> np.ndarray:
    """Adaptive Gaussian threshold: ambang dihitung per-lingkungan lokal
    sehingga tahan terhadap gradien cahaya."""
    block = block if block % 2 == 1 else block + 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def estimate_skew(gray: np.ndarray, max_angle: float = 30.0) -> float:
    """Estimasi sudut kemiringan dengan Probabilistic Hough Transform.

    Garis yang dicari: tepi kemasan/bingkai label dan garis dasar (baseline)
    baris teks. Sudut akhir = median berbobot panjang garis, hanya untuk garis
    mendekati horizontal (|sudut| < max_angle).
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    W = gray.shape[1]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80,
                            minLineLength=int(W * 0.12), maxLineGap=12)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        if abs(ang) < max_angle:
            angles.append(ang)
            weights.append(np.hypot(dx, dy))
    if len(angles) < 2:
        return 0.0
    order = np.argsort(angles)
    a, wts = np.array(angles)[order], np.array(weights)[order]
    cum = np.cumsum(wts)
    return float(a[np.searchsorted(cum, cum[-1] / 2)])  # median berbobot


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(img_bgr: np.ndarray, cfg: PreprocessConfig) -> dict:
    """Menjalankan seluruh tahap pra-pemrosesan.

    Return dict berisi citra setiap tahap: color, gray, clahe, denoised,
    binary, serta skew_angle (derajat).
    """
    color, _ = resize_to_width(img_bgr, cfg.target_width)
    gray = to_gray(color)

    angle = 0.0
    if cfg.deskew:
        angle = estimate_skew(gray)
        if 0.5 < abs(angle) < 30:
            color = rotate(color, angle)
            gray = to_gray(color)
        else:
            angle = 0.0

    # Reduksi noise dilakukan SEBELUM CLAHE: bila dibalik, CLAHE ikut
    # memperkuat noise sensor/JPEG sehingga tepi huruf tertutup bintik.
    denoised = reduce_noise(gray, cfg.denoise)
    clahe = equalize_contrast(denoised, cfg.clahe_clip, cfg.clahe_grid) if cfg.use_clahe else denoised
    binary = adaptive_binarize(clahe, cfg.adaptive_block, cfg.adaptive_c)
    return {"color": color, "gray": gray, "denoised": denoised, "clahe": clahe,
            "enhanced": clahe, "binary": binary, "skew_angle": angle}


def sharpness(gray: np.ndarray) -> float:
    """Varians Laplacian sebagai ukuran ketajaman (dipakai mode kamera)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
