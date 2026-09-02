"""
Halaman 5: Dashboard Satker Dekonsentrasi & Tugas Pembantuan
----------------------------------------------------------------
Memakai sumber data yang SAMA dengan Halaman 1 (data/pagu_realisasi.csv.gz), difilter
hanya ke satker dengan kewenangan "Dekonsentrasi" (DK) atau "Tugas Pembantuan" (TP) --
lihat common.py::klasifikasi_kewenangan.

Catatan penting: kolom KEWENANGAN belum ada di file data/pagu_realisasi.csv.gz saat ini
(hanya ditambahkan lewat build_from_combined_csv.py / data_prep.py versi terbaru). Kalau
data belum di-rebuild dari file sumber yang memuat info kewenangan, halaman ini akan
menampilkan pesan penjelasan alih-alih tabel kosong yang membingungkan.

Akses: satker biasa hanya melihat datanya sendiri (kalau kewenangannya memang DK/TP);
super user melihat semua satker DK/TP.
"""

import numpy as np
import pandas as pd
import streamlit as st

from common import get_data, tanggal_update_data, fmt_satker, kpi_card, klasifikasi_kewenangan

df_semua = get_data()

# Login sudah ditangani app.py (router) -- halaman ini pun cuma dimasukkan ke navigasi
# kalau kolom KEWENANGAN tersedia (lihat app.py), tapi kita cek ulang di sini sbg proteksi
# tambahan (jaga-jaga kalau ada yang coba akses langsung via URL).
auth = st.session_state.auth
is_super = auth["role"] == "super"
SCOPE_KDSATKER = None if is_super else auth["kdsatker"]

st.title("🏢 Dashboard Satker Dekonsentrasi & Tugas Pembantuan")
st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data()}")

if "KEWENANGAN" not in df_semua.columns or (df_semua["KEWENANGAN"].astype(str).str.strip() == "").all():
    st.info(
        "Kolom **kewenangan** (Kantor Pusat/Kantor Daerah/Dekonsentrasi/Tugas Pembantuan/"
        "Urusan Bersama) belum tersedia di data saat ini, jadi Halaman 5 belum bisa "
        "menampilkan daftar satker Dekonsentrasi & Tugas Pembantuan. Untuk mengaktifkan "
        "halaman ini, regenerasi `data/pagu_realisasi.csv.gz` dari file sumber yang memuat "
        "kolom kewenangan (lihat `build_from_combined_csv.py` atau `data_prep.py` -- kolom "
        "`KEWENANGAN` sudah ditangani otomatis kalau ada di file sumbernya)."
    )
    st.stop()

df_semua = df_semua.copy()
df_semua["KEWENANGAN_LABEL"] = df_semua["KEWENANGAN"].apply(klasifikasi_kewenangan)
df = df_semua[df_semua["KEWENANGAN_LABEL"].isin(["Dekonsentrasi", "Tugas Pembantuan"])]

if SCOPE_KDSATKER is not None:
    df = df[df["KDSATKER"] == SCOPE_KDSATKER]

if df.empty:
    if SCOPE_KDSATKER is not None:
        st.info("Satker Anda bukan satker dengan kewenangan Dekonsentrasi atau Tugas Pembantuan.")
    else:
        st.warning("Tidak ada satker dengan kewenangan Dekonsentrasi atau Tugas Pembantuan di data ini.")
    st.stop()


# --------------------------------------------------------------------------
# Sidebar - filter
# --------------------------------------------------------------------------

st.sidebar.header("Filter")

tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
tahun = st.sidebar.selectbox("Tahun", tahun_list)
df_tahun = df[df["TAHUN"] == tahun]

satker_options = (
    df_tahun[["KDSATKER", "NMSATKER"]]
    .drop_duplicates()
    .sort_values("KDSATKER")
)
satker_options["LABEL"] = satker_options["KDSATKER"].apply(fmt_satker) + " - " + satker_options["NMSATKER"]

if SCOPE_KDSATKER is not None:
    # Satker biasa: tidak perlu dropdown, datanya memang sudah dibatasi ke satkernya sendiri.
    satker_pilih_label = satker_options["LABEL"].tolist()
    df_satker_filter = df_tahun
else:
    SEMUA_SATKER = "— Semua Satker —"
    satker_pilih_label = st.sidebar.multiselect(
        "Satker", satker_options["LABEL"].tolist(), default=[],
        placeholder="Semua satker (kosongkan = tampilkan semua)",
    )
    if satker_pilih_label:
        kdsatker_pilih = satker_options.loc[
            satker_options["LABEL"].isin(satker_pilih_label), "KDSATKER"
        ]
        df_satker_filter = df_tahun[df_tahun["KDSATKER"].isin(kdsatker_pilih)]
    else:
        df_satker_filter = df_tahun

jenis_options = (
    df_tahun[["JENIS BELANJA", "LABEL_JENIS_BELANJA"]]
    .drop_duplicates()
    .sort_values("JENIS BELANJA")
)
jenis_pilih = st.sidebar.multiselect(
    "Jenis Belanja", jenis_options["LABEL_JENIS_BELANJA"].tolist(), default=[],
    placeholder="Semua jenis belanja (kosongkan = tampilkan semua)",
)
if jenis_pilih:
    df_final = df_satker_filter[df_satker_filter["LABEL_JENIS_BELANJA"].isin(jenis_pilih)]
else:
    df_final = df_satker_filter


# --------------------------------------------------------------------------
# KPI ringkas
# --------------------------------------------------------------------------

pagu_total = df_final["PAGU"].sum()
realisasi_total = df_final["REALISASI"].sum()
blokir_total = df_final["BLOKIR"].sum() if "BLOKIR" in df_final.columns else 0
persen_total = (realisasi_total / pagu_total * 100) if pagu_total else 0
jumlah_satker = df_final["KDSATKER"].nunique()

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Jumlah Satker", f"{jumlah_satker:,}")
with k2:
    kpi_card("Total Pagu", f"Rp {pagu_total:,.0f}")
with k3:
    kpi_card("Total Realisasi", f"Rp {realisasi_total:,.0f}", f"{persen_total:.1f}% dari pagu")
with k4:
    kpi_card("Total Blokir", f"Rp {blokir_total:,.0f}")

st.divider()


# --------------------------------------------------------------------------
# Tabel: kode satker, nama satker, kewenangan, pagu, realisasi, blokir, % realisasi
# --------------------------------------------------------------------------

st.subheader("Rincian per Satker")

agg_cols = {"PAGU": "sum", "REALISASI": "sum"}
if "BLOKIR" in df_final.columns:
    agg_cols["BLOKIR"] = "sum"

tabel = (
    df_final.groupby(["KDSATKER", "NMSATKER", "KEWENANGAN_LABEL"])
    .agg(agg_cols)
    .reset_index()
)
if "BLOKIR" not in tabel.columns:
    tabel["BLOKIR"] = 0.0

tabel["Persentase Realisasi"] = (
    tabel["REALISASI"] / tabel["PAGU"].replace(0, np.nan) * 100
).fillna(0)

tabel["Kode Satker"] = tabel["KDSATKER"].apply(fmt_satker)
tabel = tabel.rename(columns={
    "NMSATKER": "Nama Satker",
    "KEWENANGAN_LABEL": "Kewenangan",
    "PAGU": "Pagu",
    "REALISASI": "Realisasi",
    "BLOKIR": "Blokir",
})
tabel = tabel[["Kode Satker", "Nama Satker", "Kewenangan", "Pagu", "Realisasi", "Blokir", "Persentase Realisasi"]]
tabel = tabel.sort_values("Pagu", ascending=False)

baris_total = pd.DataFrame([{
    "Kode Satker": "", "Nama Satker": "TOTAL", "Kewenangan": "",
    "Pagu": tabel["Pagu"].sum(),
    "Realisasi": tabel["Realisasi"].sum(),
    "Blokir": tabel["Blokir"].sum(),
    "Persentase Realisasi": (
        tabel["Realisasi"].sum() / tabel["Pagu"].sum() * 100
    ) if tabel["Pagu"].sum() else 0,
}])
tabel_tampil = pd.concat([tabel, baris_total], ignore_index=True)


def _tebalkan_total(row):
    return ["font-weight: bold;" if row["Nama Satker"] == "TOTAL" else "" for _ in row]


st.dataframe(
    tabel_tampil.style
    .apply(_tebalkan_total, axis=1)
    .format({
        "Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}", "Blokir": "Rp {:,.0f}",
        "Persentase Realisasi": "{:.1f}%",
    }),
    use_container_width=True,
    hide_index=True,
)
