"""Menggabungkan seluruh tahap menjadi satu pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import cv2
import numpy as np

from .config import AppConfig
from .info_extractor import ProductInfo, build_speech, extract_info
from .ocr_engine import OCROutput, ocr_rois
from .preprocessing import preprocess
from .text_localization import localize_text


@dataclass
class PipelineResult:
    stages: dict
    boxes: list
    ocr: OCROutput
    info: ProductInfo
    speech: str
    timings: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.ocr.text


class TextReaderPipeline:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()

    def process(self, img_bgr: np.ndarray, today: date | None = None, mode: str = "ringkas") -> PipelineResult:
        t = {}
        t0 = time.perf_counter()
        stages = preprocess(img_bgr, self.cfg.pre)
        t["preprocess"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        loc = localize_text(stages["enhanced"], self.cfg.loc)
        stages["edges"], stages["mask"] = loc["edges"], loc["mask"]
        t["localization"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        ocr = ocr_rois(stages["enhanced"], loc["boxes"], self.cfg.ocr, self.cfg.loc.pad)
        t["ocr"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        info = extract_info(ocr.lines, today)
        speech = build_speech(info, ocr.text, mode)
        t["extraction"] = time.perf_counter() - t3
        t["total"] = time.perf_counter() - t0
        return PipelineResult(stages, loc["boxes"], ocr, info, speech, t)


def draw_boxes(img: np.ndarray, boxes, color=(0, 200, 0)) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, b in enumerate(boxes):
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
        cv2.putText(out, str(i + 1), (b.x, max(12, b.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def stage_montage(result: PipelineResult, width: int = 420) -> np.ndarray:
    """Gabungkan citra tiap tahap menjadi satu grid (untuk debug / laporan)."""
    s = result.stages
    tiles = [("1. Input", s["color"]), ("2. Grayscale", s["gray"]), ("3. Denoise (bilateral)", s["denoised"]),
             ("4. CLAHE", s["clahe"]), ("5. Adaptive Thr.", s["binary"]),
             ("6. Morph. Gradient", s["edges"]), ("7. Closing Mask", s["mask"]),
             ("8. Text ROI", draw_boxes(s["color"], result.boxes))]
    rendered = []
    for name, im in tiles:
        im = im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h = int(im.shape[0] * width / im.shape[1])
        im = cv2.resize(im, (width, h))
        bar = np.full((28, width, 3), 255, np.uint8)
        cv2.putText(bar, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(np.vstack([bar, im]))
    hmax = max(r.shape[0] for r in rendered)
    rendered = [cv2.copyMakeBorder(r, 0, hmax - r.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                for r in rendered]
    rows = [np.hstack(rendered[i:i + 4]) for i in range(0, len(rendered), 4)]
    return np.vstack(rows)
