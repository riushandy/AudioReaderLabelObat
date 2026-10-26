"""Utilitas kamera: pembukaan webcam, analisis kualitas frame, pemilihan frame tertajam.

Perbaikan terhadap versi lama (main.py):
  * analisis dilakukan pada lebar 1280 px, bukan 640 px: pada 640 px jumlah blok
    teks sering < 3 sehingga auto-capture hampir tidak pernah aktif;
  * ketajaman dihitung di area tengah (tempat kemasan) dan dinormalisasi,
    bukan dari seluruh frame termasuk latar;
  * "kemasan terlalu jauh" dideteksi dari lebar kumpulan blok teks. Pada uji
    simulasi, akurasi tanggal turun 8/8 -> 6/8 -> 3/8 saat label mengecil
    dari 100% -> 55% -> 40% lebar frame;
  * frame yang dibaca = yang tertajam dari beberapa frame terakhir, bukan
    sekadar frame saat ini.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CameraConfig, LocalizationConfig
from .preprocessing import resize_to_width, to_gray
from .text_localization import TextBox, localize_text


def open_camera(source, cfg: CameraConfig, is_file: bool = False) -> cv2.VideoCapture:
    """Buka webcam dengan MJPG + resolusi tinggi + autofocus. File video dibuka apa adanya."""
    if is_file:
        return cv2.VideoCapture(source)
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)     # MSMF sering mengabaikan resolusi/fokus
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # sebelum resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    for _ in range(15):                                  # beri waktu autofocus/exposure menyesuaikan
        cap.read()
    return cap


def center_crop(gray: np.ndarray, frac: float = 0.7) -> np.ndarray:
    h, w = gray.shape[:2]
    dh, dw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    return gray[dh:h - dh, dw:w - dw]


def center_sharpness(gray: np.ndarray, norm_width: int = 640) -> float:
    """Varians Laplacian area tengah, dinormalisasi lebar -> sebanding antar resolusi kamera."""
    roi = center_crop(gray)
    s = norm_width / roi.shape[1]
    roi = cv2.resize(roi, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def text_cluster(boxes: list[TextBox], W: int, H: int):
    """Kotak pembatas kumpulan blok teks utama (buang pencilan jauh dari pusat massa)."""
    if len(boxes) < 2:
        return None
    cx = np.array([b.x + b.w / 2 for b in boxes])
    cy = np.array([b.y + b.h / 2 for b in boxes])
    mx, my = np.median(cx), np.median(cy)
    keep = [b for b in boxes if abs(b.x + b.w / 2 - mx) < 0.35 * W and abs(b.y + b.h / 2 - my) < 0.35 * H]
    if len(keep) < 2:
        return None
    return (min(b.x for b in keep), min(b.y for b in keep),
            max(b.x + b.w for b in keep), max(b.y + b.h for b in keep))


@dataclass
class FrameStatus:
    hint: str | None            # petunjuk suara; None = siap dibaca
    kind: str | None            # dark | glare | notext | far | motion | blur
    brightness: float
    glare: float
    sharpness: float
    motion: float
    n_boxes: int
    cluster_frac: float         # lebar kumpulan teks / lebar frame
    boxes: list = field(default_factory=list)   # koordinat pada gambar analisis
    scale: float = 1.0          # frame asli -> gambar analisis
    small_gray: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.hint is None


def analyse(frame: np.ndarray, prev_small: np.ndarray | None,
            cam: CameraConfig, loc: LocalizationConfig) -> FrameStatus:
    work = frame if frame.shape[1] <= cam.analysis_width else resize_to_width(frame, cam.analysis_width)[0]
    scale = work.shape[1] / frame.shape[1]
    gray = to_gray(work)
    roi = center_crop(gray)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    bright = float(roi.mean())
    glare = float((roi >= 252).mean())
    sharp = center_sharpness(gray)
    motion = 0.0 if prev_small is None else float(cv2.absdiff(small, prev_small).mean())
    boxes = localize_text(cv2.GaussianBlur(gray, (3, 3), 0), loc)["boxes"]
    cl = text_cluster(boxes, work.shape[1], work.shape[0])
    frac = (cl[2] - cl[0]) / work.shape[1] if cl else 0.0

    hint = kind = None
    if bright < cam.dark_threshold:
        hint = "Cahaya terlalu gelap. Pindah ke tempat yang lebih terang."; kind = "dark"
    elif glare > cam.glare_frac:
        hint = "Terlalu silau. Miringkan kemasan sedikit."; kind = "glare"
    elif bright > cam.bright_threshold:
        hint = "Terlalu terang. Jauhi jendela atau lampu di belakang."; kind = "bright"
    elif len(boxes) < cam.min_text_boxes:
        hint = "Teks belum terdeteksi. Arahkan sisi kemasan yang bertulisan ke kamera."; kind = "notext"
    elif frac < cam.min_cluster_frac:
        hint = "Kemasan terlalu jauh. Dekatkan sedikit."; kind = "far"
    elif motion > cam.motion_threshold:
        hint = "Tahan kemasan lebih stabil."; kind = "motion"
    elif sharp < cam.sharpness_threshold:
        hint = "Gambar buram. Jauhkan sedikit kemasan dan tahan stabil."; kind = "blur"
    return FrameStatus(hint, kind, bright, glare, sharp, motion, len(boxes), frac, boxes, scale, small)


class FrameRing:
    """Simpan N frame terakhir beserta ketajamannya; ambil yang tertajam saat capture."""

    def __init__(self, size: int = 8):
        self._d: deque = deque(maxlen=size)

    def push(self, frame: np.ndarray, sharp: float):
        self._d.append((sharp, frame))

    def best(self, k: int = 1) -> list[np.ndarray]:
        return [f for _, f in sorted(self._d, key=lambda t: t[0], reverse=True)[:k]]
