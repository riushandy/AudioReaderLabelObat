"""Tahap 3 - Ekstraksi karakter dengan Tesseract (PyTesseract).

Setiap ROI hasil lokalisasi:
  potong -> Otsu threshold lokal -> koreksi polaritas (teks hitam di latar
  putih) -> upscale ke tinggi standar -> tambah border -> Tesseract
  (--psm 7 untuk satu baris, --psm 6 untuk blok).
Kata dengan confidence rendah dibuang agar TTS tidak membacakan "sampah".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract

from .config import OCRConfig
from .text_localization import TextBox, crop


@dataclass
class OCRLine:
    text: str
    conf: float
    box: TextBox | None = None
    font_h: int = 0            # tinggi kotak (proksi ukuran huruf)


@dataclass
class OCROutput:
    lines: list[OCRLine] = field(default_factory=list)
    used_fallback: bool = False

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def mean_conf(self) -> float:
        return float(np.mean([l.conf for l in self.lines])) if self.lines else 0.0


def available_langs(requested: str) -> str:
    """Pakai bahasa yang benar-benar terpasang (fallback ke eng)."""
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    langs = [l for l in requested.split("+") if l in installed]
    return "+".join(langs) if langs else "eng"


def normalize_polarity(bin_img: np.ndarray) -> np.ndarray:
    """Pastikan teks hitam di atas latar putih. Latar ditaksir dari piksel tepi ROI."""
    border = np.concatenate([bin_img[0, :], bin_img[-1, :], bin_img[:, 0], bin_img[:, -1]])
    return cv2.bitwise_not(bin_img) if border.mean() < 127 else bin_img


def prepare_roi(gray_roi: np.ndarray, target_h: int) -> np.ndarray:
    """Siapkan ROI untuk Tesseract.

    Otsu dipakai HANYA untuk menentukan polaritas (teks gelap/terang). Citra
    yang diberikan ke Tesseract tetap grayscale ter-upscale: hasil uji
    menunjukkan binarisasi internal Tesseract pada ROI grayscale lebih akurat
    daripada citra yang sudah kami binarisasi (lihat laporan, bagian Evaluasi).
    """
    h, w = gray_roi.shape[:2]
    scale = max(1.0, target_h / float(h))
    if scale > 1.0:
        gray_roi = cv2.resize(gray_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(cv2.GaussianBlur(gray_roi, (3, 3), 0), 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if normalize_polarity(bw) is not bw:      # latar gelap -> balik
        gray_roi = cv2.bitwise_not(gray_roi)
    return cv2.copyMakeBorder(gray_roi, 12, 12, 12, 12, cv2.BORDER_REPLICATE)


def prepare_roi_binary(gray_roi: np.ndarray, target_h: int) -> np.ndarray:
    """Varian lama (ROI dibinarisasi Otsu sendiri) - disimpan untuk perbandingan."""
    h, w = gray_roi.shape[:2]
    scale = max(1.0, target_h / float(h))
    if scale > 1.0:
        gray_roi = cv2.resize(gray_roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, bw = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = normalize_polarity(bw)
    return cv2.copyMakeBorder(bw, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


_GARBAGE = re.compile(r"^[^A-Za-z0-9]+$")


def _clean_line(words: list[tuple[str, float]], min_conf: float) -> tuple[str, float]:
    kept = [(w, c) for w, c in words if c >= min_conf and w.strip() and not _GARBAGE.match(w)]
    if not kept:
        return "", 0.0
    text = " ".join(w for w, _ in kept)
    return text, float(np.mean([c for _, c in kept]))


def ocr_image(img: np.ndarray, lang: str, psm: int, oem: int, min_conf: float) -> list[tuple[str, float]]:
    """OCR satu citra -> list (teks_baris, conf). Mengelompokkan kata per baris Tesseract."""
    cfg = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
    lines: dict[tuple, list] = {}
    for i, word in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((word, conf))
    out = []
    for key in sorted(lines):
        t, c = _clean_line(lines[key], min_conf)
        if t:
            out.append((t, c))
    return out


def ocr_rois(gray: np.ndarray, boxes: list[TextBox], cfg: OCRConfig, pad: int = 6) -> OCROutput:
    lang = available_langs(cfg.lang)
    result = OCROutput()
    for box in boxes:
        prep = prepare_roi_binary if cfg.roi_binarize else prepare_roi
        roi = prep(crop(gray, box, pad), cfg.roi_target_h)
        psm = 7 if box.w / max(box.h, 1) > 2.5 else 6
        for text, conf in ocr_image(roi, lang, psm, cfg.oem, cfg.min_word_conf):
            if sum(ch.isalnum() for ch in text) >= 2:
                result.lines.append(OCRLine(text, conf, box, box.h))

    if cfg.fallback_full_page and len(result.text) < cfg.min_chars_before_fallback:
        # ROI gagal (mis. teks sangat rapat) -> OCR seluruh halaman, segmentasi otomatis
        full = ocr_image(gray, lang, 3, cfg.oem, cfg.min_word_conf)
        if len("".join(t for t, _ in full)) > len(result.text):
            result = OCROutput([OCRLine(t, c) for t, c in full], used_fallback=True)
    return result


def ocr_raw(img: np.ndarray, lang: str = "ind+eng") -> str:
    """Baseline: Tesseract langsung pada citra asli tanpa pra-pemrosesan."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    return pytesseract.image_to_string(rgb, lang=available_langs(lang), config="--oem 1 --psm 3")
