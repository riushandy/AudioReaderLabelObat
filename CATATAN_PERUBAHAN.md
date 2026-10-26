# Perubahan (timpa file berikut di repo; file lain tidak berubah)

| File | Status | Isi |
|---|---|---|
| `main.py` | ubah | CameraReader ditulis ulang: kamera 1080p MJPG + autofocus, analisis 1280 px, panduan "dekatkan", ambil frame tertajam, baca frame tambahan bila tanggal belum yakin, batas waktu agar tidak menunggu selamanya. Argumen baru `--width/--height`. |
| `src/camera.py` | baru | open_camera, analyse (kecerahan, silau, ketajaman area tengah, gerakan, blok teks, lebar kumpulan teks), FrameRing |
| `src/verify.py` | baru | verify_expiry (baca ulang baris tanggal 3 skala + whitelist + voting), fix_units, fuzzy_words |
| `src/pipeline.py` | ubah | memanggil verify; method baru `process_frames` + voting tanggal antar-frame |
| `src/info_extractor.py` | ubah | `expiry_confident`; ucapan "kurang jelas" bila tanggal tak pasti; nama produk menghindari baris nama perusahaan |
| `src/config.py` | ubah | CameraConfig baru (resolusi, ambang, dll.), OCRConfig.verify_expiry / verify_scales |

Menjalankan: `python main.py --camera 0`. Overlay menampilkan tajam / gerak / terang / blok / lebar(%) untuk menyetel ambang di `CameraConfig`.
