"""
build_from_combined_csv.py
----------------------------
Memproses satu file gabungan pagu_real21-26.csv (2021-2026 dalam 1 file) menjadi
data/pagu_realisasi.csv yang dipakai app.py.

Karakteristik file sumber yang perlu ditangani:
- Separator kolom: titik-koma (;), bukan koma
- Encoding: latin-1 (bukan utf-8 -- ada karakter non-utf8)
- Format angka gaya Indonesia: titik sebagai pemisah ribuan (mis. "5.526.896.000")
- Sebagian kecil angka sangat besar tersimpan dalam notasi ilmiah dengan koma
  sebagai desimal (mis. "1,44E+11") -- kemungkinan akibat Excel saat menyimpan CSV
- Ada ~900 baris kosong di akhir file (dibuang)
- Kolom "jenis belanja" (2 digit) dan "Realisasi"/"Sisa Pagu" sudah tersedia langsung
  di file sumber, tapi tetap dihitung ulang dari JAN..DES untuk konsistensi
  (lihat catatan validasi data sebelumnya).
"""

import pandas as pd

BULAN_SRC = ["jan", "feb", "mar", "apr", "mei", "jun",
             "jul", "ags", "sep", "okt", "nov", "des"]
BULAN_OUT = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN",
             "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]

LABEL_JENIS_BELANJA = {
    51: "Belanja Pegawai", 52: "Belanja Barang", 53: "Belanja Modal",
    54: "Belanja Pembayaran Bunga Utang", 55: "Belanja Subsidi",
    56: "Belanja Hibah", 57: "Belanja Bantuan Sosial", 58: "Belanja Lain-lain",
    61: "Transfer Daerah - Dana Bagi Hasil", 62: "Transfer Daerah - Dana Alokasi Umum",
    63: "Transfer Daerah - Dana Alokasi Khusus", 64: "Transfer Daerah - Insentif Fiskal",
    65: "Transfer Daerah - Dana BOK", 66: "Transfer Daerah - Dana Desa",
}

SRC_PATH = "/mnt/user-data/uploads/pagu_real21-26.csv"


def parse_angka(series: pd.Series) -> pd.Series:
    """Parse angka format Indonesia (titik ribuan), termasuk beberapa kasus notasi
    ilmiah dengan koma desimal (mis. '1,44E+11')."""
    s = series.astype(str).str.strip()
    sci_mask = s.str.contains(r"[eE]", regex=True, na=False) & s.str.contains(",", na=False)

    hasil = pd.Series(0.0, index=s.index)
    hasil[sci_mask] = pd.to_numeric(
        s[sci_mask].str.replace(",", ".", regex=False), errors="coerce"
    )
    hasil[~sci_mask] = pd.to_numeric(
        s[~sci_mask].str.replace(".", "", regex=False), errors="coerce"
    )
    return hasil.fillna(0)


def main():
    print("Membaca file gabungan...")
    df = pd.read_csv(SRC_PATH, sep=";", encoding="latin-1", low_memory=False)
    print(f"  Dibaca: {len(df):,} baris mentah")

    df = df.dropna(subset=["TAHUN"]).copy()
    print(f"  Setelah buang baris kosong di akhir file: {len(df):,} baris")

    out = pd.DataFrame()
    out["KDDEPT"] = pd.to_numeric(df["kementerian_kode"], errors="coerce")
    out["TAHUN"] = pd.to_numeric(df["TAHUN"], errors="coerce")
    out["NMDEPT"] = df["kementerian_uraian"].astype(str).str.strip()
    out["KDSATKER"] = pd.to_numeric(df["satker_kode"], errors="coerce")
    out["NMSATKER"] = df["satker_uraian"].astype(str).str.strip()
    out["JENIS BELANJA"] = pd.to_numeric(df["jenis belanja"], errors="coerce")

    out["PAGU"] = parse_angka(df["pagu_dipa"])
    for src, dst in zip(BULAN_SRC, BULAN_OUT):
        out[dst] = parse_angka(df[src])
    out["BLOKIR"] = parse_angka(df["blokir"])

    # Kolom deskriptif tambahan -- dipakai fitur pencarian tematik AI (mis. "ketahanan
    # pangan di Riau", "kebakaran hutan"), bukan untuk ditampilkan di dropdown dashboard.
    out["PROVINSI"] = df["provinsi_uraian"].astype(str).str.strip()
    out["KABKOTA"] = df["kabkota_uraian"].astype(str).str.strip()
    out["FUNGSI"] = df["fungsi_uraian"].astype(str).str.strip()
    out["SUBFUNGSI"] = df["subfungsi_uraian"].astype(str).str.strip()
    out["PROGRAM"] = df["program_uraian"].astype(str).str.strip()
    out["KEGIATAN"] = df["kegiatan_uraian"].astype(str).str.strip()
    out["OUTPUT"] = df["outputkro_uraian"].astype(str).str.strip()
    out["AKUN"] = df["akun_uraian"].astype(str).str.strip()

    # Kolom kewenangan (KP/KD/DK/TP/UB -- Kantor Pusat/Kantor Daerah/Dekonsentrasi/Tugas
    # Pembantuan/Urusan Bersama), dipakai Halaman 5 (Dashboard Satker Dekon & TP). Nama
    # kolom di file sumber bisa berbeda-beda tergantung ekspor Sintesa/SPAN, jadi dicoba
    # beberapa kemungkinan nama; kalau tidak ada satupun, kolom diisi kosong (Halaman 5
    # akan menampilkan pesan kalau kolom ini kosong).
    _kandidat_kolom_kewenangan = [
        "kewenangan_uraian", "kewenangan", "kdkewenangan_uraian",
        "kewenangan_kode_uraian", "uraian_kewenangan",
    ]
    _kolom_kewenangan_ditemukan = next(
        (c for c in _kandidat_kolom_kewenangan if c in df.columns), None
    )
    if _kolom_kewenangan_ditemukan:
        out["KEWENANGAN"] = df[_kolom_kewenangan_ditemukan].astype(str).str.strip()
        print(f"  Kolom kewenangan ditemukan: '{_kolom_kewenangan_ditemukan}'")
    else:
        out["KEWENANGAN"] = ""
        print(
            "  PERINGATAN: kolom kewenangan (DK/TP/dst.) tidak ditemukan di file sumber -- "
            "Halaman 5 (Dashboard Satker Dekon & TP) tidak akan bisa menampilkan data. Kalau "
            "file sumbermu punya kolom ini dengan nama lain, tambahkan ke "
            "_kandidat_kolom_kewenangan di script ini."
        )

    before = len(out)
    out = out.dropna(subset=["KDDEPT", "TAHUN", "KDSATKER", "JENIS BELANJA"])
    dropped = before - len(out)
    if dropped:
        print(f"  {dropped:,} baris dibuang (kode dept/tahun/satker/jenis belanja tidak valid)")

    out["KDDEPT"] = out["KDDEPT"].astype(int)
    out["TAHUN"] = out["TAHUN"].astype(int)
    out["KDSATKER"] = out["KDSATKER"].astype(int)
    out["JENIS BELANJA"] = out["JENIS BELANJA"].astype(int)

    out["LABEL_JENIS_BELANJA"] = out["JENIS BELANJA"].map(LABEL_JENIS_BELANJA)
    out["LABEL_JENIS_BELANJA"] = out["LABEL_JENIS_BELANJA"].fillna(
        "Lainnya (" + out["JENIS BELANJA"].astype(str) + ")"
    )

    # REALISASI & SISA PAGU dihitung ulang dari JAN..DES (bukan dipakai langsung dari
    # kolom sumber) supaya konsisten dengan rincian bulanannya.
    out["REALISASI"] = out[BULAN_OUT].sum(axis=1)
    out["SISA PAGU"] = out["PAGU"] - out["REALISASI"]

    out.to_csv("data/pagu_realisasi.csv.gz", index=False, compression="gzip")

    print(f"\nSelesai. Total {len(out):,} baris disimpan ke data/pagu_realisasi.csv.gz")
    print("\nRingkasan per tahun:")
    print(
        out.groupby("TAHUN").agg(
            baris=("PAGU", "size"),
            total_pagu=("PAGU", "sum"),
            total_realisasi=("REALISASI", "sum"),
        )
    )


if __name__ == "__main__":
    main()
