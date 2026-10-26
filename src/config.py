"""Konfigurasi global pipeline. Semua parameter bisa diubah di satu tempat."""
from dataclasses import dataclass, field


@dataclass
class PreprocessConfig:
    target_width: int = 1600          # gambar di-resize ke lebar ini (upscale/downscale)
    clahe_clip: float = 2.0           # batas kontras CLAHE
    clahe_grid: int = 8               # ukuran tile CLAHE
    denoise: str = "nlmeans"          # "nlmeans" | "bilateral" | "median" | "none"
    adaptive_block: int = 31          # ukuran blok adaptive threshold (ganjil)
    adaptive_c: int = 12              # konstanta pengurang adaptive threshold
    deskew: bool = True               # koreksi kemiringan teks
    use_clahe: bool = True            # aktif/nonaktif CLAHE (untuk ablation study)


@dataclass
class LocalizationConfig:
    grad_kernel: tuple = (3, 3)       # kernel morphological gradient
    close_kernel_ratio: float = 0.018  # lebar kernel closing relatif terhadap lebar gambar
    close_kernel_h: int = 3           # tinggi kernel closing (piksel)
    min_box_h: int = 10               # tinggi minimum blok teks
    max_box_h_ratio: float = 0.30     # tinggi maksimum relatif terhadap tinggi gambar
    min_box_w: int = 18               # lebar minimum blok teks
    min_aspect: float = 1.0           # rasio w/h minimum (teks cenderung melebar)
    min_fill: float = 0.30            # rasio piksel tepi di dalam kotak
    pad: int = 6                      # padding ROI saat dipotong
    remove_lines: bool = True         # hapus garis bingkai dengan Hough Transform


@dataclass
class OCRConfig:
    lang: str = "ind+eng"             # bahasa Tesseract (perlu ind.traineddata)
    oem: int = 1                      # 1 = LSTM engine
    roi_target_h: int = 64            # tinggi ROI setelah di-upscale (piksel)
    min_word_conf: float = 35.0       # kata dengan confidence di bawah ini dibuang
    roi_binarize: bool = False        # True = ROI di-Otsu sendiri (varian lama, lebih buruk)
    fallback_full_page: bool = True   # OCR seluruh gambar bila ROI kurang hasil
    min_chars_before_fallback: int = 15
    verify_expiry: bool = True        # baca ulang baris tanggal + voting (anti salah-baca digit)
    verify_scales: tuple = (48, 80, 112)   # tinggi ROI (px) untuk pembacaan ulang tanggal


@dataclass
class TTSConfig:
    engine: str = "pyttsx3"           # "pyttsx3" (offline) | "gtts" (online) | "none"
    lang: str = "id"
    rate: int = 150                   # kecepatan bicara pyttsx3 (kata/menit)


@dataclass
class CameraConfig:
    width: int = 1920                   # resolusi permintaan ke webcam (fallback otomatis bila tak didukung)
    height: int = 1080
    analysis_width: int = 1280          # lebar frame untuk analisis teks (640 terlalu kecil: blok teks hilang)
    sharpness_threshold: float = 60.0   # varians Laplacian area tengah (dinormalisasi lebar 640 px)
    stable_frames: int = 10             # jumlah frame siap berturut-turut sebelum auto-capture
    motion_threshold: float = 6.0       # selisih rata-rata antar frame (gambar 160x90)
    min_text_boxes: int = 3             # minimal blok teks agar auto-capture
    min_cluster_frac: float = 0.35      # lebar kumpulan teks / lebar frame; di bawah ini = "dekatkan kemasan"
    dark_threshold: float = 70.0        # rata-rata intensitas area tengah di bawah ini = terlalu gelap
    glare_frac: float = 0.25            # porsi piksel >= 252 di area tengah; di atas ini = silau
    guidance_interval_s: float = 6.0    # jeda antar panduan suara
    ring_size: int = 8                  # jumlah frame terakhir yang disimpan untuk memilih yang tertajam
    force_after_s: float = 8.0          # bila hanya 'buram'/'jauh' selama ini, tetap baca (agar tidak menunggu selamanya)
    max_extra_frames: int = 2           # frame tambahan dibaca bila tanggal belum yakin


@dataclass
class AppConfig:
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    loc: LocalizationConfig = field(default_factory=LocalizationConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    cam: CameraConfig = field(default_factory=CameraConfig)
