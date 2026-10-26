"""Verifikasi tanggal kedaluwarsa + koreksi teks pasca-OCR.

Masalah yang dijawab (laporan Bab 7.4 no. 2): satu digit tahun salah baca
("2028" -> "2023") membuat sistem keliru menyatakan produk kedaluwarsa.

Cara kerja verify_expiry: baris yang berisi tanggal dibaca ULANG pada beberapa
skala dengan whitelist karakter tanggal (digit, huruf bulan, / . - :). Hasilnya
divoting bersama bacaan awal:
  - >= 3 dari 4 bacaan sepakat -> tanggal "yakin" (dan dikoreksi bila beda
    dari bacaan awal)
  - selain itu -> tanggal ditandai TIDAK yakin; build_speech lalu berkata
    "kurang jelas" alih-alih menyatakan aman atau kedaluwarsa.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

import pytesseract

from .config import OCRConfig
from .info_extractor import ProductInfo, parse_date
from .ocr_engine import available_langs, prepare_roi
from .text_localization import crop

_WL = "0123456789/.-:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _find_line(lines, date_):
    for ln in lines:
        d, _ = parse_date(ln.text)
        if d == date_:
            return ln
    return None


def verify_expiry(gray, ocr, info: ProductInfo, cfg: OCRConfig, pad: int = 6) -> None:
    """Mengubah `info` di tempat: expiry_date (bila dikoreksi) & expiry_confident."""
    if info.expiry_date is None:
        return
    orig = info.expiry_date
    line = _find_line(ocr.lines, orig)
    if line is None or line.box is None:
        # tanggal tersusun dari beberapa baris / hasil OCR satu halaman: tak bisa diverifikasi
        info.expiry_confident = ocr.mean_conf >= 75
        return
    lang = available_langs(cfg.lang)
    roi_src = crop(gray, line.box, pad)
    votes = [orig]
    for th in cfg.verify_scales:
        roi = prepare_roi(roi_src, th)
        try:
            txt = pytesseract.image_to_string(
                roi, lang=lang,
                config=f"--oem {cfg.oem} --psm 7 -c tessedit_char_whitelist={_WL}", timeout=5)
        except (RuntimeError, pytesseract.TesseractError):
            continue
        d, _ = parse_date(txt)
        if d:
            votes.append(d)
    top, n = Counter(votes).most_common(1)[0]
    need = min(3, len(cfg.verify_scales) + 1)
    if n >= need:
        info.expiry_date = top
        info.expiry_confident = True
        if top != orig:                       # koreksi: bacaan awal kalah suara
            info.expiry_raw = f"{info.expiry_raw} -> {top.isoformat()}"
    else:
        info.expiry_confident = False


# ---------- koreksi satuan & kata (pasca-OCR) ----------
_VOCAB = """diminum sesudah sebelum makan sehari tablet kapsul kaplet sendok takar dewasa
anak tahun kocok dahulu simpan suhu kedaluwarsa kadaluarsa komposisi indikasi aturan pakai
netto berat bersih kandungan efek samping peringatan jangan digunakan dikunyah larutkan
dalam segelas air hangat sirup suspensi tetes salep minum pagi siang malam""".split()


def fix_units(text: str) -> str:
    """'120 mi' -> '120 ml', '500 mq' -> '500 mg' (hanya setelah angka)."""
    text = re.sub(r"(\d)\s*(?:mi|m1|rnl)\b", r"\1 ml", text)
    return re.sub(r"(\d)\s*(?:mq|rng|nng|m9)\b", r"\1 mg", text)


def fuzzy_words(text: str, cutoff: float = 0.85) -> str:
    """Perbaiki kata mirip-kamus ('Oiminum' -> 'Diminum'). Hanya kata >= 5 huruf.
    Pakai pada teks aturan pakai, BUKAN pada nama produk."""
    vset = set(_VOCAB)

    def repl(m):
        w, lw = m.group(0), m.group(0).lower()
        if lw in vset:
            return w
        c = difflib.get_close_matches(lw, _VOCAB, n=1, cutoff=cutoff)
        if not c:
            return w
        b = c[0]
        return b.upper() if w.isupper() else (b.capitalize() if w[:1].isupper() else b)

    return re.sub(r"[A-Za-z]{5,}", repl, text)
