"""Tahap 4 - Ekstraksi informasi penting dari teks OCR.

Pengguna tunanetra tidak butuh seluruh teks dibacakan; yang paling penting:
nama produk, kekuatan/komposisi, tanggal kedaluwarsa (beserta statusnya),
aturan pakai/dosis, peringatan, dan nomor registrasi BPOM.
Ekstraksi memakai aturan (regex) + koreksi kesalahan OCR umum pada angka.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

MONTHS = {
    "JAN": 1, "JANUARI": 1, "JANUARY": 1,
    "FEB": 2, "FEBRUARI": 2, "FEBRUARY": 2, "PEB": 2,
    "MAR": 3, "MARET": 3, "MARCH": 3,
    "APR": 4, "APRIL": 4,
    "MEI": 5, "MAY": 5,
    "JUN": 6, "JUNI": 6, "JUNE": 6,
    "JUL": 7, "JULI": 7, "JULY": 7,
    "AGU": 8, "AGS": 8, "AGT": 8, "AUG": 8, "AGUSTUS": 8, "AUGUST": 8,
    "SEP": 9, "SEPT": 9, "SEPTEMBER": 9,
    "OKT": 10, "OCT": 10, "OKTOBER": 10, "OCTOBER": 10,
    "NOV": 11, "NOP": 11, "NOVEMBER": 11, "NOPEMBER": 11,
    "DES": 12, "DEC": 12, "DESEMBER": 12, "DECEMBER": 12,
}
MONTH_NAMES_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
                  "Agustus", "September", "Oktober", "November", "Desember"]

EXP_KEYS = (r"EXP(?:IRY|IRED)?(?:\s*DATE)?|E\s*\.?\s*D\b|ED\b|KEDALUWARSA|KADALUARSA|KADALUWARSA|"
            r"BAIK\s+DIGUNAKAN\s+SEBELUM|BEST\s+BEFORE(?:\s+END)?|BBE?\b|USE\s+BY|TGL\s*\.?\s*KEDALUWARSA")
MFG_KEYS = r"MFG|MFD|PROD(?:UKSI)?|DIBUAT|TGL\s*\.?\s*PRODUKSI|MANUFACTURED"
MON_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

DATE_PATTERNS = [
    # 31/12/2027, 31-12-27, 31.12.2027
    (re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\b"), "dmy"),
    # 31 DES 2027 / 31 DESEMBER 2027
    (re.compile(rf"\b(\d{{1,2}})\s*[\s\-/.]?\s*({MON_RE})\s*[\s\-/.]?\s*(\d{{2,4}})\b", re.I), "dMy"),
    # DES 2027 / DESEMBER 2027
    (re.compile(rf"\b({MON_RE})\s*[\s\-/.]?\s*(\d{{2,4}})\b", re.I), "My"),
    # 12/2027 atau 12-27
    (re.compile(r"\b(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b"), "my"),
]


@dataclass
class ProductInfo:
    product_name: str = ""
    strengths: list[str] = field(default_factory=list)
    expiry_raw: str = ""
    expiry_date: date | None = None
    days_left: int | None = None
    dosage: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    registration: str = ""
    netto: str = ""

    @property
    def expired(self) -> bool | None:
        return None if self.days_left is None else self.days_left < 0


def fix_digits(s: str) -> str:
    """Koreksi salah baca umum Tesseract di konteks angka: O->0, I/l->1, S->5, B->8."""
    table = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                           "S": "5", "B": "8", "Z": "2"})

    def repl(m):
        return m.group(0).translate(table)
    # hanya token yang didominasi angka & pemisah tanggal
    return re.sub(r"\b[0-9OoDIl|SBZ]{1,4}(?:[/\-.][0-9OoDIl|SBZ]{1,4}){1,2}\b",
                  lambda m: repl(m) if sum(c.isdigit() for c in m.group(0)) >= 2 else m.group(0), s)


def _year(y: str) -> int:
    y = int(y)
    return 2000 + y if y < 100 else y


def parse_date(text: str) -> tuple[date | None, str]:
    text = fix_digits(text)
    for pat, kind in DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            if kind == "dmy":
                d, mo, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
            elif kind == "dMy":
                d, mo, y = int(m.group(1)), MONTHS[m.group(2).upper()], _year(m.group(3))
            elif kind == "My":
                mo, y = MONTHS[m.group(1).upper()], _year(m.group(2))
                d = calendar.monthrange(y, mo)[1]
            else:  # my -> akhir bulan
                mo, y = int(m.group(1)), _year(m.group(2))
                d = calendar.monthrange(y, mo)[1]
            if 2010 <= y <= 2100 and 1 <= mo <= 12:
                return date(y, mo, d), m.group(0)
        except (ValueError, KeyError):
            continue
    return None, ""


def extract_expiry(lines: list[str], today: date) -> tuple[date | None, str]:
    key_re = re.compile(rf"({EXP_KEYS})\s*[:.\-]?\s*(.*)", re.I)
    for i, line in enumerate(lines):
        m = key_re.search(line)
        if not m:
            continue
        # tanggal biasanya setelah kata kunci, atau di baris berikutnya
        for cand in (m.group(2), lines[i + 1] if i + 1 < len(lines) else ""):
            d, raw = parse_date(cand)
            if d:
                return d, raw
    # tanpa kata kunci: ambil tanggal paling akhir yang bukan tanggal produksi
    mfg_re = re.compile(MFG_KEYS, re.I)
    found = []
    for line in lines:
        if mfg_re.search(line):
            continue
        d, raw = parse_date(line)
        if d and d >= date(today.year - 5, 1, 1):
            found.append((d, raw))
    return max(found) if found else (None, "")


DOSAGE_RE = re.compile(
    r"(aturan\s+pakai|dosis|cara\s+pakai|sehari|tiap|setiap|\d\s*[x×]\s*\d|\d\s*[x×]\s*sehari|"
    r"diminum|sendok|tablet|kaplet|kapsul|sesudah\s+makan|sebelum\s+makan|dewasa|anak)", re.I)
WARN_RE = re.compile(r"(peringatan|perhatian|awas|jangan|hindari|tidak\s+boleh|simpan|alergi|"
                     r"mengandung|contains|hamil|menyusui)", re.I)
STRENGTH_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|IU|%)(?:\s*/\s*\d*\s*(?:ml|mL|tablet|kapsul))?", re.I)
REG_RE = re.compile(r"\b(?:BPOM\s*RI\s*)?(?:MD|ML|DBL|DTL|DKL|GBL|GKL|SD|SI|TR|QD|NA)\s*[0-9OIlSB ]{6,16}[A-Z0-9]{0,3}")
PACK_RE = re.compile(r"^\s*(?:\d+\s+)?(?:tablet|kaplet|kapsul|sachet|sirup|strip)\b.*?(?:\d+\s*(?:ml)?)?\s*$|^\s*\d+\s+(?:tablet|kaplet|kapsul|sachet)", re.I)
NETTO_RE = re.compile(r"(netto|neto|berat\s+bersih|isi\s+bersih|isi|net\s*wt)\s*[:.]?\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|gram|kg|ml|mL|l|liter|tablet|kaplet|kapsul|sachet)\b)", re.I)


def extract_info(ocr_lines, today: date | None = None) -> ProductInfo:
    """ocr_lines: list[OCRLine] (punya .text dan .font_h) atau list[str]."""
    today = today or date.today()
    texts = [l if isinstance(l, str) else l.text for l in ocr_lines]
    heights = [0 if isinstance(l, str) else l.font_h for l in ocr_lines]
    info = ProductInfo()

    # Nama produk: baris dengan huruf terbesar (kotak tertinggi) yang berisi kata
    cands = [(h, t) for t, h in zip(texts, heights)
             if sum(c.isalpha() for c in t) >= 3 and not re.search(EXP_KEYS, t, re.I)]
    if cands:
        if any(h for h, _ in cands):
            info.product_name = max(cands, key=lambda x: x[0])[1]
        else:
            info.product_name = cands[0][1]

    info.expiry_date, info.expiry_raw = extract_expiry(texts, today)
    if info.expiry_date:
        info.days_left = (info.expiry_date - today).days

    joined = "\n".join(texts)
    info.strengths = list(dict.fromkeys(m.group(0).strip() for m in STRENGTH_RE.finditer(joined)))[:4]
    m = REG_RE.search(joined)
    info.registration = m.group(0).strip() if m else ""
    m = NETTO_RE.search(joined)
    info.netto = m.group(2) if m else ""

    for t in texts:
        if PACK_RE.search(t) and len(t) < 30:      # mis. "10 Tablet", "Sirup 60 ml"
            info.netto = info.netto or t
            continue
        if DOSAGE_RE.search(t) and t not in info.dosage and t != info.product_name:
            info.dosage.append(t)
        elif WARN_RE.search(t) and t not in info.warnings:
            info.warnings.append(t)
    info.dosage = info.dosage[:4]
    info.warnings = info.warnings[:3]
    return info


def format_date_id(d: date) -> str:
    return f"{d.day} {MONTH_NAMES_ID[d.month]} {d.year}"


def build_speech(info: ProductInfo, full_text: str = "", mode: str = "ringkas") -> str:
    """Susun kalimat yang akan diucapkan. mode: 'ringkas' | 'lengkap'."""
    parts = []
    if info.product_name:
        parts.append(f"Produk: {info.product_name}.")
    if info.strengths:
        parts.append("Kandungan: " + ", ".join(info.strengths) + ".")
    if info.expiry_date:
        tgl = format_date_id(info.expiry_date)
        if info.expired:
            parts.append(f"Peringatan! Produk sudah kedaluwarsa sejak {tgl}. Jangan digunakan.")
        elif info.days_left is not None and info.days_left <= 30:
            parts.append(f"Perhatian, produk akan kedaluwarsa pada {tgl}, sisa {info.days_left} hari.")
        else:
            parts.append(f"Kedaluwarsa: {tgl}. Produk masih dapat digunakan.")
    else:
        parts.append("Tanggal kedaluwarsa tidak terbaca.")
    if info.dosage:
        items = [re.sub(r"^\s*(aturan\s+pakai|dosis|cara\s+pakai)\s*[:.]?\s*", "", d, flags=re.I) for d in info.dosage]
        parts.append("Aturan pakai: " + ". ".join(i for i in items if i) + ".")
    if info.netto:
        parts.append(f"Isi: {info.netto}.")
    if mode == "lengkap":
        if info.warnings:
            parts.append("Peringatan pada kemasan: " + ". ".join(info.warnings) + ".")
        if full_text:
            parts.append("Teks lengkap: " + full_text.replace("\n", ". "))
    return " ".join(parts)
