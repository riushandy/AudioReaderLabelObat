"""Tahap 5 - Text-to-Speech.

Dua backend:
  * pyttsx3 : offline (SAPI5 di Windows, NSSpeech di macOS, eSpeak di Linux).
  * gTTS    : online (Google), suara Bahasa Indonesia lebih natural.
Bicara dijalankan di thread terpisah agar tampilan kamera tidak membeku.
Jika backend gagal, teks tetap dicetak ke konsol (tidak crash).
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading

from .config import TTSConfig


class Speaker:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.engine_name = cfg.engine
        self._q: queue.Queue[str | None] = queue.Queue()
        self._busy = threading.Event()
        self._thread = None
        if self.engine_name != "none":
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ---------- backend ----------
    def _init_pyttsx3(self):
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", self.cfg.rate)
        # pilih suara Bahasa Indonesia bila tersedia
        for v in eng.getProperty("voices"):
            ident = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
            if "indones" in ident or "id_id" in ident or "id-id" in ident or ident.endswith("/id"):
                eng.setProperty("voice", v.id)
                break
        return eng

    def _speak_gtts(self, text: str):
        from gtts import gTTS
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        gTTS(text, lang=self.cfg.lang).save(path)
        try:
            from playsound import playsound  # opsional
            playsound(path)
        except Exception:
            print(f"[TTS] audio disimpan di {path} (pasang 'playsound' untuk memutar otomatis)")

    def _worker(self):
        # pyttsx3: buat engine BARU untuk tiap ucapan. Memakai satu engine untuk
        # runAndWait() berulang di thread membuat suara hanya keluar sekali
        # (bug umum SAPI5/Windows dan NSSpeech/macOS).
        if self.engine_name == "pyttsx3":
            try:
                self._init_pyttsx3().stop()
            except Exception as e:
                print(f"[TTS] pyttsx3 tidak tersedia ({e}); hanya mencetak teks.")
                self.engine_name = "print"
        while True:
            text = self._q.get()
            if text is None:
                break
            self._busy.set()
            try:
                if self.engine_name == "pyttsx3":
                    eng = self._init_pyttsx3()
                    eng.say(text)
                    eng.runAndWait()
                    eng.stop()
                    del eng
                elif self.engine_name == "gtts":
                    self._speak_gtts(text)
            except Exception as e:
                print(f"[TTS] gagal bicara: {e}")
            finally:
                self._busy.clear()

    # ---------- API ----------
    def say(self, text: str, interrupt: bool = False):
        print(f"[SUARA] {text}")
        if self._thread is None:
            return
        if interrupt:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        self._q.put(text)

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def close(self):
        if self._thread is not None:
            self._q.put(None)


_FILE_ENGINE = None


def save_speech(text: str, path: str, engine: str = "pyttsx3", lang: str = "id", rate: int = 150) -> str:
    """Simpan ucapan ke file audio (untuk demo / pengujian)."""
    if engine == "gtts":
        from gtts import gTTS
        gTTS(text, lang=lang).save(path)
        return path
    global _FILE_ENGINE
    import pyttsx3
    if _FILE_ENGINE is None:
        # referensi global: driver eSpeak pyttsx3 memakai weakref sehingga engine
        # lokal bisa terhapus GC di tengah sintesis teks panjang (ReferenceError)
        _FILE_ENGINE = pyttsx3.init()
    eng = _FILE_ENGINE
    eng.setProperty("rate", rate)
    for v in eng.getProperty("voices"):
        ident = f"{v.id} {v.name}".lower()
        if "indones" in ident or v.id.lower().endswith("/id"):
            eng.setProperty("voice", v.id)
            break
    path = os.path.abspath(path)   # driver eSpeak butuh path absolut
    try:
        eng.save_to_file(text, path)
        eng.runAndWait()
    except Exception as e:
        print(f"[TTS] pyttsx3 gagal menyimpan audio: {e}")
    if not os.path.exists(path):
        # cadangan untuk Linux: driver eSpeak pyttsx3 kadang gagal pada teks panjang
        import shutil
        import subprocess
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if exe:
            subprocess.run([exe, "-v", "id", "-s", str(rate), "-w", path, text], check=False)
    return path