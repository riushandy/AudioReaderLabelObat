"""Evaluasi kuantitatif pipeline pada dataset ber-ground-truth.

Metrik:
  * CER  (Character Error Rate)  = edit_distance(prediksi, GT) / panjang GT
  * WER  (Word Error Rate)       = edit_distance kata / jumlah kata GT
  * Word Recall                  = proporsi kata GT yang muncul di hasil OCR
                                   (tidak bergantung urutan baca)
  * Expiry Accuracy              = tanggal kedaluwarsa terekstrak == GT
  * Name Found                   = nama produk terbaca (kemiripan >= 85%)
  * Waktu proses per gambar

Varian yang dibandingkan:
  baseline_raw   : Tesseract langsung pada citra asli (tanpa OpenCV)
  gray_otsu      : grayscale + Otsu global lalu Tesseract (pra-proses sederhana)
  pipeline       : sistem usulan (denoise + CLAHE + deskew + lokalisasi ROI + OCR per-ROI)
  ablation_*     : pipeline dengan satu komponen dimatikan

Pemakaian:
  python tools/evaluate.py --data data/samples --out outputs/eval
  python tools/evaluate.py --data data/samples --quick   # tanpa ablation
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import time
from datetime import date

import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import AppConfig  # noqa: E402
from src.info_extractor import extract_info  # noqa: E402
from src.ocr_engine import available_langs, ocr_raw  # noqa: E402
from src.pipeline import TextReaderPipeline  # noqa: E402

TODAY = date(2026, 10, 18)  # tanggal acuan tetap agar hasil dapat direproduksi


def norm(s: str) -> str:
    s = s.upper().replace("\n", " ")
    s = re.sub(r"[^A-Z0-9/.,:%\- ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(pred: str, gt: str) -> float:
    p, g = norm(pred), norm(gt)
    return min(1.0, Levenshtein.distance(p, g) / max(1, len(g)))


def wer(pred: str, gt: str) -> float:
    p, g = norm(pred).split(), norm(gt).split()
    return min(1.0, Levenshtein.distance(p, g) / max(1, len(g)))


def word_recall(pred: str, gt: str) -> float:
    pw = set(norm(pred).split())
    gw = [w for w in norm(gt).split() if len(w) > 1]
    return sum(w in pw for w in gw) / max(1, len(gw))


def name_found(pred: str, name: str) -> bool:
    return fuzz.partial_ratio(norm(name), norm(pred)) >= 85


def run_variant(name, items, data_dir, cfg=None):
    rows = []
    pipe = TextReaderPipeline(cfg) if cfg else None
    lang = available_langs("ind+eng")
    for it in items:
        img = cv2.imread(os.path.join(data_dir, it["file"]))
        gt_text = "\n".join(it["lines"])
        t0 = time.perf_counter()
        if name == "baseline_raw":
            text = ocr_raw(img, lang)
            lines = [l for l in text.splitlines() if l.strip()]
            info = extract_info(lines, TODAY)
        elif name == "gray_otsu":
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(bw, lang=lang, config="--oem 1 --psm 3")
            lines = [l for l in text.splitlines() if l.strip()]
            info = extract_info(lines, TODAY)
        else:
            res = pipe.process(img, today=TODAY)
            text, info = res.text, res.info
        dt = time.perf_counter() - t0
        rows.append({
            "variant": name, "file": it["file"], "level": it["level"], "category": it["category"],
            "dark_bg": it.get("dark_background", False),
            "cer": cer(text, gt_text), "wer": wer(text, gt_text), "recall": word_recall(text, gt_text),
            "expiry_ok": info.expiry_date is not None and info.expiry_date.isoformat() == it["expiry"],
            "expiry_pred": info.expiry_date.isoformat() if info.expiry_date else "",
            "expiry_gt": it["expiry"], "name_ok": name_found(text, it["product_name"]),
            "time_s": dt, "pred_text": text,
        })
        print(f"  [{name}] {it['file']:<22} CER={rows[-1]['cer']:.3f} recall={rows[-1]['recall']:.2f} "
              f"exp={'OK' if rows[-1]['expiry_ok'] else 'X'} t={dt:.2f}s")
    return rows


def summarize(rows):
    out = {}
    for key in sorted({(r["variant"]) for r in rows}):
        vr = [r for r in rows if r["variant"] == key]
        out[key] = {"all": _agg(vr)}
        for lvl in ("easy", "medium", "hard"):
            out[key][lvl] = _agg([r for r in vr if r["level"] == lvl])
    return out


def _agg(rs):
    if not rs:
        return {}
    return {"n": len(rs),
            "cer": float(np.mean([r["cer"] for r in rs])),
            "wer": float(np.mean([r["wer"] for r in rs])),
            "recall": float(np.mean([r["recall"] for r in rs])),
            "expiry_acc": float(np.mean([r["expiry_ok"] for r in rs])),
            "name_acc": float(np.mean([r["name_ok"] for r in rs])),
            "time_s": float(np.mean([r["time_s"] for r in rs]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--quick", action="store_true", help="lewati ablation study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(os.path.join(args.data, "ground_truth.json"), encoding="utf-8"))

    variants = [("baseline_raw", None), ("gray_otsu", None), ("pipeline", AppConfig())]
    if not args.quick:
        for vname, mutate in [
            ("ablation_no_denoise", lambda c: setattr(c.pre, "denoise", "none")),
            ("ablation_no_clahe", lambda c: setattr(c.pre, "use_clahe", False)),
            ("ablation_no_deskew", lambda c: setattr(c.pre, "deskew", False)),
            ("ablation_no_line_removal", lambda c: setattr(c.loc, "remove_lines", False)),
            ("ablation_roi_binarized", lambda c: setattr(c.ocr, "roi_binarize", True)),
            ("fast_mode", lambda c: (setattr(c.pre, "denoise", "bilateral"), setattr(c.pre, "target_width", 1280))),
        ]:
            c = copy.deepcopy(AppConfig())
            mutate(c)
            variants.append((vname, c))

    rows = []
    for name, cfg in variants:
        print(f"== {name} ==")
        rows += run_variant(name, items, args.data, cfg)

    with open(os.path.join(args.out, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "pred_text"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "pred_text"})
    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump([{k: r[k] for k in ("variant", "file", "pred_text")} for r in rows], f, indent=1, ensure_ascii=False)
    summary = summarize(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=========== RINGKASAN ===========")
    print(f"{'varian':<26}{'CER':>7}{'WER':>7}{'Recall':>8}{'ExpAcc':>8}{'Name':>7}{'t(s)':>7}")
    for k, v in summary.items():
        a = v["all"]
        print(f"{k:<26}{a['cer']:>7.3f}{a['wer']:>7.3f}{a['recall']:>8.3f}{a['expiry_acc']:>8.2f}"
              f"{a['name_acc']:>7.2f}{a['time_s']:>7.2f}")


if __name__ == "__main__":
    main()
