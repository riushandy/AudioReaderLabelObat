# OCR & Audio Reader Label Obat dan Makanan Kemasan

Pembaca teks otomatis berbasis kamera untuk penyandang tunanetra dan low vision.
Sistem membaca label kemasan obat/makanan dengan OpenCV + Tesseract OCR, mengambil
informasi penting (nama produk, kandungan, **tanggal kedaluwarsa beserta statusnya**,
aturan pakai), lalu membacakannya dengan Text-to-Speech Bahasa Indonesia.

Mata kuliah Teknologi Berkembang (B) - OpenCV, Assignment #1 (Week 8).

## Alur sistem

```
Kamera / gambar
  -> [1] Pra-pemrosesan  : resize 1600px, grayscale, deskew (Hough), denoise (NL-Means),
                           CLAHE, adaptive threshold
  -> [2] Lokalisasi teks : morphological gradient, Otsu, hapus garis panjang (Hough),
                           closing horizontal, findContours + filter geometri -> ROI baris teks
  -> [3] OCR             : per ROI -> koreksi polaritas -> upscale -> Tesseract LSTM (ind+eng)
  -> [4] Ekstraksi info  : regex nama/kandungan/kedaluwarsa/dosis/BPOM + koreksi salah baca angka
  -> [5] Text-to-Speech  : pyttsx3 (offline) atau gTTS (online)
```

## Struktur folder

```
main.py                    aplikasi utama (mode gambar, kamera, video)
src/config.py              semua parameter
src/preprocessing.py       tahap 1
src/text_localization.py   tahap 2
src/ocr_engine.py          tahap 3
src/info_extractor.py      tahap 4 + penyusunan kalimat ucapan
src/tts.py                 tahap 5
src/pipeline.py            penggabung tahap + visualisasi
tools/generate_dataset.py  generator dataset uji sintetis + ground truth
tools/evaluate.py          evaluasi CER/WER/recall/akurasi kedaluwarsa + ablation
tools/plot_results.py      grafik hasil evaluasi
data/examples/             6 contoh gambar dari test set (langsung bisa dicoba)
data/samples/              ground truth dev set (gambar dibuat ulang dengan generator, seed 7)
data/test_heldout/         ground truth test set (gambar dibuat ulang dengan generator, seed 2026)
results/                   hasil evaluasi test set (summary.json, per_image.csv, grafik, contoh audio)
```

## Instalasi

1. **Python 3.10+** lalu:
   ```bash
   pip install -r requirements.txt
   ```
2. **Tesseract OCR 5.x**
   - Windows: unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki.
     Saat instalasi, centang *Additional language data -> Indonesian*.
     Jika `tesseract` tidak ada di PATH, jalankan dengan
     `--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"`.
   - Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-ind espeak-ng`
   - macOS: `brew install tesseract tesseract-lang`
   - Cek: `tesseract --list-langs` harus memuat `eng` dan `ind`.
     (Bila `ind` tidak ada, program otomatis memakai `eng` saja.)
3. **TTS**: pyttsx3 memakai suara bawaan OS (Windows SAPI5 / macOS / eSpeak di Linux).
   Untuk suara Indonesia di Windows: Settings -> Time & Language -> Speech -> Add voices -> Indonesian.
   Alternatif suara lebih natural (butuh internet): `--tts gtts`.

## Menjalankan

```bash
# Mode kamera (auto-capture saat gambar tajam & stabil + panduan suara)
python main.py --camera 0

# Mode kamera lebih cepat (bilateral filter, 1280 px) untuk laptop pelan
python main.py --camera 0 --fast

# Satu gambar, simpan citra tiap tahap untuk analisis
python main.py --image data/examples/food_12_hard.png --debug outputs/debug

# Baca lengkap (termasuk peringatan & seluruh teks), simpan audio
python main.py --image foto.jpg --mode lengkap --save-audio hasil.wav

# File video sebagai pengganti kamera, rekam tampilan (untuk video demo)
python main.py --video demo.mp4 --record outputs/demo_view.mp4
```

Tombol mode kamera: **SPASI** baca sekarang, **R** ulangi suara, **F** baca teks lengkap,
**A** auto-capture on/off, **Q/ESC** keluar.

Mode kamera memberi panduan suara: "Cahaya terlalu gelap", "Teks belum terdeteksi",
"Gambar buram, jauhkan sedikit kemasan". Pembacaan otomatis terjadi bila gambar tajam
(varians Laplacian), tidak bergerak (selisih antar-frame kecil), dan ada minimal 3 blok teks
selama 12 frame berturut-turut.

## Dataset & evaluasi

Gambar dataset tidak disertakan di ZIP agar ukurannya kecil; jalankan dua perintah generator di bawah untuk membuatnya ulang. Angka di laporan dihitung di Linux dengan font DejaVu; di Windows/macOS generator memakai Arial/Liberation sehingga hasil bisa sedikit berbeda.

```bash
python tools/generate_dataset.py --out data/samples --seed 7          # dev set
python tools/generate_dataset.py --out data/test_heldout --seed 2026  # test set
python tools/evaluate.py --data data/test_heldout --out outputs/eval_heldout
python tools/plot_results.py --summary outputs/eval_heldout/summary.json --out outputs/eval_heldout
```

Dataset sintetis berisi 12 label fiktif (8 obat, 4 makanan) x 3 tingkat kesulitan
(easy / medium / hard: rotasi, perspektif, cahaya tidak merata, blur, noise, glare, JPEG).
Merek di dataset ini fiktif. Teks ground truth diketahui pasti, sehingga CER/WER bisa dihitung.

**Foto nyata**: simpan foto kemasan asli di `data/real/` dan buat `ground_truth.json`
dengan format yang sama (`file`, `level`, `category`, `lines`, `expiry`, `product_name`),
lalu jalankan `tools/evaluate.py --data data/real`.

## Library

OpenCV, NumPy, PyTesseract + Tesseract OCR 5 (model LSTM `ind`, `eng`), pyttsx3, gTTS,
RapidFuzz (evaluasi), Pillow (generator dataset), Matplotlib (grafik).
