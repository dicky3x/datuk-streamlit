"""
build_prioritas_strategis.py
------------------------------
Memproses prioritas_presiden_2026.xlsx / program_strategis2026.xlsx (sheet dgn nama
sesuai judul dashboard, mis. "prioritas presiden 2026") menjadi data/prioritas_presiden.csv.gz
dan data/program_strategis.csv.gz yang dipakai Halaman 2 & 3.

Fleksibel thd variasi format export:
- Kalau sheet dgn nama persis tidak ketemu, otomatis pakai sheet pertama.
- Kalau kolom TAHUN tidak ada di file sumber, diisi otomatis (default 2026).
- Kalau kolom bulan tertentu (mis. sep..des) belum ada, diisi 0.
"""

import pandas as pd

BULAN_SRC = ["jan", "feb", "mar", "apr", "mei", "jun", "jul", "ags", "sep", "okt", "nov", "des"]
BULAN_OUT = ["JAN", "FEB", "MAR", "APR", "MEI", "JUN", "JUL", "AGS", "SEP", "OKT", "NOV", "DES"]

def konversi(src_path, sheet_name, kolom_kode_kategori, kolom_nama_kategori, out_path, tahun_default=2026):
    xl = pd.ExcelFile(src_path)
    sheet_dipakai = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
    if sheet_dipakai != sheet_name:
        print(f"  [!] Sheet '{sheet_name}' tidak ditemukan, pakai sheet pertama: '{sheet_dipakai}'")
    df = pd.read_excel(xl, sheet_name=sheet_dipakai)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame(index=df.index)
    if "TAHUN" in df.columns:
        out["TAHUN"] = pd.to_numeric(df["TAHUN"], errors="coerce").fillna(tahun_default).astype(int)
    else:
        out["TAHUN"] = pd.Series(tahun_default, index=df.index, dtype=int)
    out["KDDEPT"] = df["kementerian_kode"].astype(int)
    out["NMDEPT"] = df["kementerian_uraian"].astype(str).str.strip()
    out["KDSATKER"] = df["satker_kode"].astype(int)
    out["NMSATKER"] = df["satker_uraian"].astype(str).str.strip()
    out["PROVINSI"] = df["provinsi_uraian"].astype(str).str.strip()
    out["KABKOTA"] = df["kabkota_uraian"].astype(str).str.strip()
    out["FUNGSI"] = df["fungsi_uraian"].astype(str).str.strip()
    out["SUBFUNGSI"] = df["subfungsi_uraian"].astype(str).str.strip()
    out["PROGRAM"] = df["program_uraian"].astype(str).str.strip()
    out["KEGIATAN_KODE"] = df["kegiatan_kode"].astype(str).str.strip()
    out["KEGIATAN"] = df["kegiatan_uraian"].astype(str).str.strip()
    out["OUTPUT_KODE"] = df["outputkro_kode"].astype(str).str.strip()
    out["OUTPUT"] = df["outputkro_uraian"].astype(str).str.strip()
    out["SUBOUTPUT_KODE"] = df["suboutputro_kode"].astype(str).str.strip()
    out["SUBOUTPUT"] = df["suboutputro_uraian"].astype(str).str.strip()
    out["AKUN"] = df["akun_uraian"].astype(str).str.strip()
    out["KATEGORI_KODE"] = df[kolom_kode_kategori].astype(str).str.strip()
    out["KATEGORI"] = df[kolom_nama_kategori].astype(str).str.strip()
    out["PAGU"] = pd.to_numeric(df["pagu_dipa"], errors="coerce").fillna(0)
    for src, dst in zip(BULAN_SRC, BULAN_OUT):
        out[dst] = pd.to_numeric(df[src], errors="coerce").fillna(0) if src in df.columns else 0
    out["BLOKIR"] = pd.to_numeric(df.get("blokir", 0), errors="coerce").fillna(0)
    out["REALISASI"] = out[BULAN_OUT].sum(axis=1)
    out["SISA PAGU"] = out["PAGU"] - out["REALISASI"]

    out.to_csv(out_path, index=False, compression="gzip")
    print(f"Selesai: {len(out):,} baris -> {out_path}")
    print(f"  Kategori: {sorted(out['KATEGORI'].unique())}")
    print(f"  Tahun: {sorted(out['TAHUN'].unique())}")

if __name__ == "__main__":
    konversi(
        "/tmp/build_data/prioritas_presiden_2026.xlsx",
        "prioritas presiden 2026",
        "jenisprioritaspresiden_kode",
        "jenisprioritaspresiden_uraian",
        "/tmp/build_data/prioritas_presiden.csv.gz",
    )
    konversi(
        "/tmp/build_data/program_strategis2026.xlsx",
        "program strategis2026",
        "jenisprogramstrategis_kode",
        "jenisprogramstrategis_uraian",
        "/tmp/build_data/program_strategis.csv.gz",
    )
