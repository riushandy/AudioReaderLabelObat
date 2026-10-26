"""Pembaca Teks Otomatis untuk Individu dengan Gangguan Penglihatan.

Contoh pemakaian:
  python main.py --camera 0                  # mode kamera (webcam), auto-capture
  python main.py --image data/samples/med_01_easy.png --debug outputs/
  python main.py --image foto.jpg --mode lengkap --tts gtts
  python main.py --image foto.jpg --save-audio hasil.wav

Tombol pada mode kamera:
  SPASI = baca sekarang   R = ulangi suara   F = baca teks lengkap
  A     = auto-capture on/off                Q/ESC = keluar
"""
from __future__ import annotations

import argparse
import os
import threading
import time

import cv2
import numpy as np

from src.config import AppConfig
from src.camera import FrameRing, analyse, open_camera
from src.pipeline import TextReaderPipeline, draw_boxes, stage_montage
from src.preprocessing import resize_to_width
from src.text_localization import TextBox
from src.tts import Speaker, save_speech


def print_result(res):
    print("\n========== HASIL OCR ==========")
    print(res.text or "(tidak ada teks terbaca)")
    print("========== INFORMASI ==========")
    i = res.info
    print(f"Nama produk  : {i.product_name}")
    print(f"Kandungan    : {', '.join(i.strengths)}")
    print(f"Kedaluwarsa  : {i.expiry_date} (raw='{i.expiry_raw}', sisa hari={i.days_left}, "
          f"{'YAKIN' if i.expiry_confident else 'TIDAK YAKIN - ucapkan kurang jelas'})")
    print(f"Aturan pakai : {' | '.join(i.dosage)}")
    print(f"No. Reg      : {i.registration}")
    print(f"Netto        : {i.netto}")
    print("========== WAKTU (detik) ==========")
    print("  ".join(f"{k}={v:.3f}" if k != "frames" else f"frames={int(v)}" for k, v in res.timings.items()))
    print(f"\n[UCAPAN] {res.speech}\n")


def run_image(args, cfg):
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Gagal membaca gambar: {args.image}")
    pipe = TextReaderPipeline(cfg)
    res = pipe.process(img, mode=args.mode)
    print_result(res)

    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        cv2.imwrite(os.path.join(args.debug, f"{base}_stages.png"), stage_montage(res))
        cv2.imwrite(os.path.join(args.debug, f"{base}_boxes.png"), draw_boxes(res.stages["color"], res.boxes))
        with open(os.path.join(args.debug, f"{base}_ocr.txt"), "w", encoding="utf-8") as f:
            f.write(res.text + "\n\n[UCAPAN]\n" + res.speech + "\n")
        print(f"Debug disimpan di {args.debug}/")

    if args.save_audio:
        save_speech(res.speech, args.save_audio, engine=args.tts if args.tts != "none" else "pyttsx3")
        print(f"Audio disimpan: {args.save_audio}")
    elif args.tts != "none":
        spk = Speaker(cfg.tts)
        spk.say(res.speech)
        time.sleep(0.5)
        while spk.busy:
            time.sleep(0.2)
        spk.close()


class CameraReader:
    """Mode kamera dengan auto-capture dan panduan suara untuk pengguna tunanetra."""

    def __init__(self, args, cfg: AppConfig):
        self.args, self.cfg = args, cfg
        self.pipe = TextReaderPipeline(cfg)
        self.spk = Speaker(cfg.tts)
        self.auto = True
        self.processing = False
        self.last_result = None
        self.last_guidance = 0.0
        self.stable = 0
        self.prev_small = None
        self.cooldown_until = 0.0
        self.soft_since = None            # sejak kapan hanya 'buram'/'jauh' yang menghalangi

    def guide(self, msg: str):
        now = time.time()
        if now - self.last_guidance > self.cfg.cam.guidance_interval_s and not self.spk.busy:
            self.spk.say(msg)
            self.last_guidance = now

    def _process_async(self, frames):
        def job():
            try:
                res = self.pipe.process_frames(frames, mode=self.args.mode,
                                               log=lambda m: self.spk.say(m, interrupt=True))
                self.last_result = res
                print_result(res)
                if res.text.strip():
                    self.spk.say(res.speech, interrupt=True)
                else:
                    self.spk.say("Teks tidak terbaca. Putar kemasan atau dekatkan ke kamera.", interrupt=True)
            finally:
                self.processing = False
                self.cooldown_until = time.time() + 8.0
        self.processing = True
        self.spk.say("Memproses, mohon tahan.", interrupt=True)
        threading.Thread(target=job, daemon=True).start()

    def _capture(self, ring: FrameRing):
        self.stable, self.soft_since = 0, None
        self._process_async([f.copy() for f in ring.best(1 + self.cfg.cam.max_extra_frames)])

    def run(self):
        is_file = bool(self.args.video)
        source = self.args.video if is_file else self.args.camera
        cc = self.cfg.cam
        if self.args.width:
            cc.width, cc.height = self.args.width, self.args.height or int(self.args.width * 9 / 16)
        cap = open_camera(source, cc, is_file)
        if not cap.isOpened():
            raise SystemExit(f"Sumber video tidak dapat dibuka: {source}")
        if not is_file:
            print(f"[KAMERA] resolusi aktual {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        ring = FrameRing(cc.ring_size)
        self.spk.say("Pembaca label siap. Arahkan kemasan ke kamera dan tahan stabil.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            st = analyse(frame, self.prev_small, cc, self.cfg.loc)
            self.prev_small = st.small_gray
            ring.push(frame, st.sharpness)
            self.stable = self.stable + 1 if st.ready else 0

            # hanya 'buram'/'jauh' yang menghalangi: jangan menunggu selamanya
            if st.kind in ("blur", "far") and st.motion < cc.motion_threshold:
                self.soft_since = self.soft_since or time.time()
            else:
                self.soft_since = None
            force = self.soft_since is not None and time.time() - self.soft_since > cc.force_after_s

            if self.auto and not self.processing and time.time() > self.cooldown_until:
                if st.hint:
                    self.guide(st.hint)
                if self.stable >= cc.stable_frames or force:
                    self._capture(ring)

            # tampilan untuk pendamping / pengembang
            disp_w = 640
            vis, ds = resize_to_width(frame, disp_w)
            k = ds / st.scale
            vis = draw_boxes(vis, [TextBox(int(b.x * k), int(b.y * k), int(b.w * k), int(b.h * k)) for b in st.boxes])
            status = (f"tajam={st.sharpness:.0f} gerak={st.motion:.1f} terang={st.brightness:.0f} blok={st.n_boxes} "
                      f"lebar={st.cluster_frac*100:.0f}% auto={'ON' if self.auto else 'OFF'}"
                      f"{' | MEMPROSES' if self.processing else ''}")
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(vis, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, "SIAP" if st.ready else (st.hint or ""), (6, vis.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if st.ready else (0, 0, 255), 1, cv2.LINE_AA)
            if self.args.record is not None:
                self.args.record.write(cv2.resize(vis, (640, 480)))
            if self.args.no_window:
                key = 255
                if self.args.video:          # file video: tunggu proses selesai agar adil
                    while self.processing:
                        time.sleep(0.05)
            else:
                cv2.imshow("OCR Audio Reader", vis)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and not self.processing:
                self._capture(ring)
            elif key == ord("r") and self.last_result:
                self.spk.say(self.last_result.speech, interrupt=True)
            elif key == ord("f") and self.last_result:
                from src.info_extractor import build_speech
                self.spk.say(build_speech(self.last_result.info, self.last_result.text, "lengkap"), interrupt=True)
            elif key == ord("a"):
                self.auto = not self.auto
                self.spk.say("Auto aktif." if self.auto else "Auto nonaktif. Tekan spasi untuk membaca.", interrupt=True)
        while self.processing:               # tunggu pemrosesan terakhir
            time.sleep(0.1)
        cap.release()
        if not self.args.no_window:
            cv2.destroyAllWindows()
        time.sleep(0.3)
        while self.spk.busy:
            time.sleep(0.2)
        self.spk.close()


def main():
    ap = argparse.ArgumentParser(description="OCR & Audio Reader label obat / makanan kemasan")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path gambar kemasan")
    src.add_argument("--camera", type=int, help="indeks kamera (mis. 0)")
    src.add_argument("--video", help="file video sebagai pengganti kamera (uji / demo)")
    ap.add_argument("--mode", choices=["ringkas", "lengkap"], default="ringkas", help="isi ucapan")
    ap.add_argument("--tts", choices=["pyttsx3", "gtts", "none"], default="pyttsx3")
    ap.add_argument("--lang", default="ind+eng", help="bahasa Tesseract")
    ap.add_argument("--debug", help="folder untuk menyimpan citra tiap tahap")
    ap.add_argument("--save-audio", help="simpan ucapan ke file audio (.wav/.mp3)")
    ap.add_argument("--tesseract-cmd", help="path tesseract.exe (Windows)")
    ap.add_argument("--fast", action="store_true", help="mode cepat: bilateral filter + lebar 1280 px")
    ap.add_argument("--no-window", action="store_true", help="mode kamera/video tanpa jendela tampilan")
    ap.add_argument("--width", type=int, help="lebar resolusi kamera yang diminta (default 1920)")
    ap.add_argument("--height", type=int, help="tinggi resolusi kamera yang diminta (default 1080)")
    ap.add_argument("--record", help="rekam tampilan mode kamera ke file .mp4 (untuk video demo)")
    args = ap.parse_args()

    if args.tesseract_cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    cfg = AppConfig()
    cfg.tts.engine = args.tts
    cfg.ocr.lang = args.lang
    if args.fast:
        cfg.pre.denoise, cfg.pre.target_width = "bilateral", 1280
    if args.image:
        run_image(args, cfg)
    else:
        rec_path, args.record = args.record, None
        if rec_path:
            args.record = cv2.VideoWriter(rec_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
        try:
            CameraReader(args, cfg).run()
        finally:
            if args.record is not None:
                args.record.release()


if __name__ == "__main__":
    main()
