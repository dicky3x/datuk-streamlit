"""
data_prep.py
-------------
Menyiapkan data mentah (xlsx dari aplikasi SPAN/Sintesa) menjadi file CSV ramping
yang dipakai oleh app.py. Jalankan sekali setiap kali ada file sumber baru:

    python data_prep.py path/ke/file_sumber.xlsx

Hasilnya disimpan di data/pagu_realisasi.csv
"""

import sys
import pandas as pd

# Kolom yang benar-benar dipakai dashboard (kolom lain di file sumber dibuang
# supaya file tetap ringan untuk di-deploy ke GitHub/Streamlit Cloud)
KOLOM_DIPAKAI = [
    "TAHUN", "KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER",
    "JENIS BELANJA",
    "PAGU", "JAN", "FEB", "MAR", "APR", "MEI", "JUN",
    "JUL", "AGS", "SEP", "OKT", "NOV", "DES",
    "REALISASI", "SISA PAGU", "BLOKIR",
    # Kewenangan (KP/KD/DK/TP/UB) -- dipakai Halaman 5 (Dashboard Satker Dekon & TP).
    # Diambil hanya kalau ada di file sumber; kalau tidak, Halaman 5 akan menampilkan pesan.
    "KEWENANGAN",
]

# Label jenis belanja (berdasarkan kode akun 2 digit pertama pada data Sintesa/SPAN)
LABEL_JENIS_BELANJA = {
    51: "Belanja Pegawai",
    52: "Belanja Barang",
    53: "Belanja Modal",
    54: "Belanja Pembayaran Bunga Utang",
    55: "Belanja Subsidi",
    56: "Belanja Hibah",
    57: "Belanja Bantuan Sosial",
    58: "Belanja Lain-lain",
    61: "Transfer Daerah - Dana Bagi Hasil",
    62: "Transfer Daerah - Dana Alokasi Umum",
    63: "Transfer Daerah - Dana Alokasi Khusus",
    64: "Transfer Daerah - Insentif Fiskal",
    65: "Transfer Daerah - Dana BOK",
    66: "Transfer Daerah - Dana Desa",
}


def main(src_path: str, out_path: str = "data/pagu_realisasi.csv.gz"):
    df = pd.read_excel(src_path)
    df = df[[c for c in KOLOM_DIPAKAI if c in df.columns]].copy()

    # Satker & kementerian ditampilkan sebagai "KODE - NAMA"
    df["KDDEPT"] = df["KDDEPT"].astype(int)
    df["KDSATKER"] = df["KDSATKER"].astype(int)

    df["LABEL_JENIS_BELANJA"] = df["JENIS BELANJA"].map(LABEL_JENIS_BELANJA).fillna(
        "Lainnya (" + df["JENIS BELANJA"].astype(str) + ")"
    )

    df.to_csv(out_path, index=False, compression="gzip" if out_path.endswith(".gz") else None)
    print(f"Selesai. {len(df):,} baris disimpan ke {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pemakaian: python data_prep.py path/ke/file_sumber.xlsx")
        sys.exit(1)
    main(sys.argv[1])
