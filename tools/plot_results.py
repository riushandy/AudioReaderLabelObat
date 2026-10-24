"""Buat grafik dari outputs/eval/summary.json untuk laporan.

  python tools/plot_results.py --summary outputs/eval/summary.json --out outputs/eval
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # urutan kategori tetap
LABELS = {"baseline_raw": "Baseline (Tesseract langsung)",
          "gray_otsu": "Grayscale + Otsu",
          "pipeline": "Pipeline usulan"}
ABL = {"pipeline": "Pipeline lengkap",
       "ablation_no_denoise": "tanpa denoise",
       "ablation_no_clahe": "tanpa CLAHE",
       "ablation_no_deskew": "tanpa deskew",
       "ablation_no_line_removal": "tanpa hapus garis (Hough)",
       "ablation_roi_binarized": "ROI dibinarisasi sendiri",
       "fast_mode": "mode cepat (bilateral, 1280px)"}


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_levels(summary, metric, ylabel, path, pct=True):
    levels = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    w = 0.26
    for i, v in enumerate(["baseline_raw", "gray_otsu", "pipeline"]):
        vals = [summary[v][l][metric] * (100 if pct else 1) for l in levels]
        xs = [j + (i - 1) * (w + 0.02) for j in range(3)]
        bars = ax.bar(xs, vals, w, color=SERIES[i], label=LABELS[v])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(3), ["Easy", "Medium", "Hard"], color=INK)
    ax.set_ylabel(ylabel, color=INK2)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation(summary, path, level="hard"):
    keys = [k for k in ABL if k in summary]
    vals = [summary[k][level]["cer"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    ys = range(len(keys))[::-1]
    ax.barh(list(ys), vals, 0.55, color=SERIES[0])
    for y, v in zip(ys, vals):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), [ABL[k] for k in keys], color=INK, fontsize=8)
    ax.set_xlabel(f"CER pada citra tingkat '{level}' (%) - lebih kecil lebih baik", color=INK2, fontsize=8)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/eval/summary.json")
    ap.add_argument("--out", default="outputs/eval")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    os.makedirs(a.out, exist_ok=True)
    plot_levels(s, "cer", "CER (%) - lebih kecil lebih baik", os.path.join(a.out, "chart_cer_levels.png"))
    plot_levels(s, "expiry_acc", "Akurasi tgl. kedaluwarsa (%)", os.path.join(a.out, "chart_expiry_levels.png"))
    plot_ablation(s, os.path.join(a.out, "chart_ablation.png"))
    print("grafik disimpan di", a.out)


if __name__ == "__main__":
    main()
