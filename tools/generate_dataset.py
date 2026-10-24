"""Generator dataset uji sintetis label obat & makanan kemasan (dengan ground truth).

Tujuan: menyediakan data uji terkontrol dengan teks yang diketahui pasti,
sehingga akurasi OCR (CER/WER) dan ekstraksi tanggal kedaluwarsa dapat diukur
secara objektif. Merek dalam data ini FIKTIF. Dataset foto nyata dikumpulkan
terpisah (lihat README, folder data/real/).

Tiga tingkat kesulitan:
  easy   : label datar (seperti hasil scan), noise ringan
  medium : label di atas latar acak, rotasi +-6 derajat, perspektif, cahaya
           tidak merata, blur ringan
  hard   : rotasi +-12 derajat, blur lebih kuat, kontras rendah, pantulan
           (glare), kompresi JPEG, ukuran label lebih kecil

Pemakaian:
  python tools/generate_dataset.py --out data/samples --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
             "C:/Windows/Fonts", "/Library/Fonts", "/System/Library/Fonts/Supplemental"]
FONT_FILES = {
    "bold": ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"],
    "regular": ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf", "Arial.ttf"],
    "condensed": ["DejaVuSansCondensed.ttf", "LiberationSansNarrow-Regular.ttf", "arialn.ttf", "Arial.ttf"],
    "serif": ["DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "times.ttf", "Times New Roman.ttf"],
}


def font(kind: str, size: int):
    for name in FONT_FILES[kind]:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ----------------------------- konten label -----------------------------
MEDICINES = [
    ("FEVRANOL", "Paracetamol 500 mg", "10 Tablet", ["Aturan pakai: 3 x sehari 1 tablet", "Diminum sesudah makan"]),
    ("TUSOLIN", "Dekstrometorfan HBr 15 mg/5 ml", "Sirup 60 ml", ["Dewasa: 3 x sehari 2 sendok takar", "Anak 6-12 tahun: 3 x sehari 1 sendok takar"]),
    ("AMOXIVA", "Amoxicillin 500 mg", "10 Kapsul", ["Aturan pakai: 3 x sehari 1 kapsul", "Habiskan sesuai resep dokter"]),
    ("CETRIZA", "Cetirizine 10 mg", "10 Tablet", ["Dosis: 1 x sehari 1 tablet", "Dapat menyebabkan kantuk"]),
    ("MAGASOL", "Antasida Doen", "Tablet Kunyah 20", ["Aturan pakai: 3 x sehari 1-2 tablet", "Dikunyah 1 jam sebelum makan"]),
    ("VITACIX", "Vitamin C 1000 mg", "10 Tablet Effervescent", ["Dosis: 1 x sehari 1 tablet", "Larutkan dalam 200 ml air"]),
    ("IBUFEN", "Ibuprofen 400 mg", "10 Kaplet", ["Aturan pakai: 3 x sehari 1 kaplet", "Diminum sesudah makan"]),
    ("LORADIN", "Loratadine 10 mg", "10 Tablet", ["Dewasa: 1 x sehari 1 tablet", "Simpan di bawah suhu 30 C"]),
]
FOODS = [
    ("KERIPIK SINGKONG", "Rasa Balado Pedas", "Berat Bersih: 75 g", ["Komposisi: singkong, minyak nabati, garam, cabai", "Mengandung alergen: kacang"]),
    ("SUSU UHT COKELAT", "Minuman Susu Steril", "Isi Bersih: 200 ml", ["Komposisi: susu sapi segar, gula, kakao", "Simpan di tempat sejuk dan kering"]),
    ("MIE GORENG SPESIAL", "Rasa Ayam Bawang", "Berat Bersih: 85 g", ["Masak dalam 400 ml air selama 3 menit", "Mengandung gandum dan kedelai"]),
    ("BISKUIT GANDUM", "Tinggi Serat", "Berat Bersih: 120 g", ["Komposisi: tepung gandum, gula, mentega", "Mengandung susu dan telur"]),
]
MONTHS_ID = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN", "JUL", "AGU", "SEP", "OKT", "NOV", "DES"]

SCHEMES = [  # (latar, teks, band_latar, band_teks)
    ((255, 255, 255), (20, 20, 20), (0, 70, 160), (255, 255, 255)),
    ((250, 248, 235), (110, 10, 10), (180, 20, 30), (255, 255, 255)),
    ((232, 242, 255), (10, 30, 90), (10, 30, 90), (255, 255, 255)),
    ((255, 255, 255), (0, 60, 30), (0, 120, 60), (255, 255, 255)),
    ((255, 240, 210), (60, 30, 0), (230, 120, 0), (20, 20, 20)),
    ((25, 60, 120), (255, 255, 255), (255, 210, 0), (20, 20, 20)),   # teks terang di latar gelap
]


def random_expiry(rng: random.Random, fmt: int):
    year = rng.choice([2025, 2026, 2027, 2028])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    if fmt == 0:
        s = f"{month:02d}/{year}"
        import calendar
        day = calendar.monthrange(year, month)[1]
    elif fmt == 1:
        s = f"{day:02d}/{month:02d}/{year}"
    elif fmt == 2:
        s = f"{day:02d} {MONTHS_ID[month - 1]} {year}"
    else:
        s = f"{MONTHS_ID[month - 1]} {year}"
        import calendar
        day = calendar.monthrange(year, month)[1]
    return s, f"{year:04d}-{month:02d}-{day:02d}"


def label_content(rng: random.Random, idx: int, kind: str):
    if kind == "med":
        brand, generic, pack, dosage = MEDICINES[idx % len(MEDICINES)]
        exp_s, exp_iso = random_expiry(rng, idx % 4)
        key = rng.choice(["Exp. Date", "EXP", "ED", "Kedaluwarsa"])
        reg = f"Reg. No. DBL{rng.randint(10**9, 10**10 - 1)}A1"
        lines = [(brand, "bold", 64, "band"), (generic, "bold", 34, None), (pack, "regular", 28, None)]
        lines += [(d, "regular", 24, None) for d in dosage]
        lines += [(reg, "condensed", 22, None), (f"Batch: B{rng.randint(10, 99)}K{rng.randint(10, 99)}", "condensed", 22, None),
                  (f"{key}: {exp_s}", "bold", 26, None)]
    else:
        brand, flavor, net, extra = FOODS[idx % len(FOODS)]
        exp_s, exp_iso = random_expiry(rng, (idx + 2) % 4)
        key = rng.choice(["Baik digunakan sebelum", "Best Before", "EXP"])
        reg = f"BPOM RI MD {rng.randint(10**11, 10**12 - 1)}"
        lines = [(brand, "bold", 52, "band"), (flavor, "serif", 32, None), (net, "bold", 28, None)]
        lines += [(e, "regular", 22, None) for e in extra]
        lines += [(reg, "condensed", 22, None), (f"{key}: {exp_s}", "bold", 26, None)]
    return lines, exp_iso


def render_label(lines, scheme, width=900):
    bg, fg, band_bg, band_fg = scheme
    pad, gap = 36, 14
    heights = []
    for text, kind, size, style in lines:
        f = font(kind, size)
        bbox = f.getbbox(text)
        heights.append(bbox[3] - bbox[1] + (30 if style == "band" else 0))
    H = pad * 2 + sum(heights) + gap * (len(lines) - 1)
    img = Image.new("RGB", (width, H), bg)
    d = ImageDraw.Draw(img)
    y = pad
    for (text, kind, size, style), h in zip(lines, heights):
        f = font(kind, size)
        bbox = f.getbbox(text)
        if style == "band":
            d.rectangle([0, y - 6, width, y + h - 6], fill=band_bg)
            d.text((pad, y + 9 - bbox[1]), text, font=f, fill=band_fg)
        else:
            d.text((pad, y - bbox[1]), text, font=f, fill=fg)
        y += h + gap
    d.rectangle([4, 4, width - 5, H - 5], outline=fg, width=3)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ----------------------------- degradasi -----------------------------
def cluttered_background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = rng.integers(60, 200, size=3)
    bg = np.ones((h, w, 3), np.float32) * base
    noise = cv2.resize(rng.normal(0, 25, (h // 16 + 1, w // 16 + 1, 3)).astype(np.float32), (w, h))
    bg += noise
    for _ in range(6):  # benda acak di meja
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(bg, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(20, 120)), c, -1)
    return np.clip(cv2.GaussianBlur(bg, (0, 0), 3), 0, 255).astype(np.uint8)


def place_with_perspective(label, rng, angle_max, persp, scale, canvas=(1280, 960)):
    W, H = canvas
    lh, lw = label.shape[:2]
    s = scale * min(W / lw, H / lh)
    tw, th = lw * s, lh * s
    cx, cy = W / 2 + rng.uniform(-0.08, 0.08) * W, H / 2 + rng.uniform(-0.08, 0.08) * H
    ang = np.deg2rad(rng.uniform(-angle_max, angle_max))
    corners = np.array([[-tw / 2, -th / 2], [tw / 2, -th / 2], [tw / 2, th / 2], [-tw / 2, th / 2]])
    corners += rng.uniform(-persp, persp, corners.shape) * np.array([tw, th])
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (corners @ rot.T + [cx, cy]).astype(np.float32)
    src = np.float32([[0, 0], [lw, 0], [lw, lh], [0, lh]])
    M = cv2.getPerspectiveTransform(src, dst)
    bg = cluttered_background(rng, W, H)
    warped = cv2.warpPerspective(label, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((lh, lw), 255, np.uint8), M, (W, H))
    mask3 = cv2.merge([mask] * 3).astype(np.float32) / 255
    return (warped * mask3 + bg * (1 - mask3)).astype(np.uint8)


def uneven_light(img, rng, strength):
    h, w = img.shape[:2]
    gx = np.linspace(-1, 1, w)[None, :] * rng.uniform(-1, 1)
    gy = np.linspace(-1, 1, h)[:, None] * rng.uniform(-1, 1)
    g = 1 - strength * (gx + gy + 1) / 3
    return np.clip(img.astype(np.float32) * g[..., None], 0, 255).astype(np.uint8)


def glare(img, rng, alpha):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)),
                (int(w * 0.12), int(h * 0.05)), rng.uniform(0, 180), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 25)[..., None] * alpha
    return np.clip(img * (1 - m) + 255 * m, 0, 255).astype(np.uint8)


def degrade(label, level, rng):
    if level == "easy":
        img = label.copy()
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
        return img
    if level == "medium":
        img = place_with_perspective(label, rng, angle_max=6, persp=0.03, scale=0.85)
        img = uneven_light(img, rng, 0.35)
        img = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = np.clip(img + rng.normal(0, 7, img.shape), 0, 255).astype(np.uint8)
        return img
    # hard
    img = place_with_perspective(label, rng, angle_max=12, persp=0.05, scale=0.7)
    img = uneven_light(img, rng, 0.5)
    img = glare(img, rng, 0.55)
    img = (img.astype(np.float32) * 0.65 + 50).astype(np.uint8)  # kontras rendah
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 45])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    gt = []
    items = [("med", i) for i in range(len(MEDICINES))] + [("food", i) for i in range(len(FOODS))]
    for n, (kind, idx) in enumerate(items, 1):
        lines, exp_iso = label_content(rng_py, idx, kind)
        scheme = SCHEMES[n % len(SCHEMES)]
        label = render_label(lines, scheme)
        for level in ("easy", "medium", "hard"):
            name = f"{kind}_{n:02d}_{level}.png"
            cv2.imwrite(os.path.join(args.out, name), degrade(label, level, rng_np))
            gt.append({"file": name, "category": kind, "level": level,
                       "lines": [l[0] for l in lines], "expiry": exp_iso,
                       "product_name": lines[0][0], "dark_background": scheme[0][0] < 100})
    with open(os.path.join(args.out, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    print(f"{len(gt)} gambar + ground_truth.json disimpan di {args.out}")


if __name__ == "__main__":
    main()
