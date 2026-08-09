"""
Halaman 1: Dashboard Pagu & Realisasi Satker
----------------------------------------------
Streamlit + Groq (narasi & chat AI). Supabase opsional untuk sumber data terpusat
(lihat README.md, bagian "Pakai Supabase sebagai sumber data").

Kode yang dipakai bersama halaman lain (loading data, login, proyeksi, dst) ada di
common.py -- lihat file itu untuk detail implementasinya.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BULAN_KOLOM, BULAN_LABEL, fmt_satker, fmt_dept,
    get_data, tanggal_update_data, kpi_card,
    hitung_proyeksi_agregat, hitung_proyeksi_per_kategori, isi_tabel_proyeksi,
    hitung_bulan_penuh_terakhir, buat_cari_generik, render_ai_section, render_dataset_upload_qa,
    LABEL_BELANJA_PEGAWAI,
)

df = get_data()

# Login sudah ditangani app.py (router) sebelum halaman ini dipanggil -- auth dijamin ada.
auth = st.session_state.auth
is_super = auth["role"] == "super"
SCOPE_KDSATKER = None if is_super else auth["kdsatker"]


# --------------------------------------------------------------------------
# Sidebar - filter
# --------------------------------------------------------------------------

st.sidebar.header("Filter")

if is_super:
    tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
    tahun = st.sidebar.selectbox("Tahun", tahun_list)

    df_tahun = df[df["TAHUN"] == tahun]

    SEMUA_DEPT = "— Semua Kementerian/Lembaga —"
    SEMUA_SATKER = "— Semua Satker —"

    dept_options = (
        df_tahun[["KDDEPT", "NMDEPT"]]
        .drop_duplicates()
        .sort_values("KDDEPT")
    )
    dept_options["LABEL"] = dept_options["KDDEPT"].apply(fmt_dept) + " - " + dept_options["NMDEPT"]
    dept_label = st.sidebar.selectbox("Kementerian/Lembaga", [SEMUA_DEPT] + dept_options["LABEL"].tolist())

    if dept_label == SEMUA_DEPT:
        kddept = None
        nmdept = "Semua Kementerian/Lembaga"
        df_dept = df_tahun
    else:
        kddept = int(dept_label.split(" - ")[0])
        nmdept = dept_options.loc[dept_options["KDDEPT"] == kddept, "NMDEPT"].iloc[0]
        df_dept = df_tahun[df_tahun["KDDEPT"] == kddept]

    satker_options = (
        df_dept[["KDSATKER", "NMSATKER"]]
        .drop_duplicates()
        .sort_values("KDSATKER")
    )
    satker_options["LABEL"] = satker_options["KDSATKER"].apply(fmt_satker) + " - " + satker_options["NMSATKER"]
    satker_label = st.sidebar.selectbox("Satuan Kerja (Satker)", [SEMUA_SATKER] + satker_options["LABEL"].tolist())

    if satker_label == SEMUA_SATKER:
        kdsatker = None
        nmsatker = "Semua Satker"
        df_satker = df_dept
    else:
        kdsatker = int(satker_label.split(" - ")[0])
        nmsatker = satker_options.loc[satker_options["KDSATKER"] == kdsatker, "NMSATKER"].iloc[0]
        df_satker = df_dept[df_dept["KDSATKER"] == kdsatker]
else:
    # User satker: tidak ada pilihan kementerian/satker -- otomatis terkunci ke satker sendiri.
    kdsatker = auth["kdsatker"]
    df_kdsatker_semua_tahun = df[df["KDSATKER"] == kdsatker]

    tahun_list = sorted(df_kdsatker_semua_tahun["TAHUN"].unique(), reverse=True)
    if not tahun_list:
        st.error(f"Tidak ada data untuk satker dengan kode {fmt_satker(kdsatker)}.")
        st.stop()
    tahun = st.sidebar.selectbox("Tahun", tahun_list)

    df_tahun = df[df["TAHUN"] == tahun]
    df_satker = df_tahun[df_tahun["KDSATKER"] == kdsatker]

    if df_satker.empty:
        st.warning(f"Satker Anda belum punya data di tahun {tahun}.")
        nmsatker = "-"
        kddept, nmdept = None, "-"
        df_dept = df_tahun.iloc[0:0]
    else:
        nmsatker = df_satker["NMSATKER"].iloc[0]
        kddept = int(df_satker["KDDEPT"].iloc[0])
        nmdept = df_satker["NMDEPT"].iloc[0]
        df_dept = df_tahun[df_tahun["KDDEPT"] == kddept]

    st.sidebar.caption(f"Satker: **{fmt_satker(kdsatker)} - {nmsatker}**")
    st.sidebar.caption(f"Kementerian/Lembaga: {fmt_dept(kddept)} - {nmdept}")


# --------------------------------------------------------------------------
# Agregasi
# --------------------------------------------------------------------------

pagu_total = df_satker["PAGU"].sum()
realisasi_total = df_satker["REALISASI"].sum()
sisa_pagu = df_satker["SISA PAGU"].sum()
persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0

monthly = df_satker[BULAN_KOLOM].sum()
kumulatif = monthly.cumsum()

bulan_terakhir, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(df_satker, tahun)

jenis_belanja = (
    df_satker.groupby("LABEL_JENIS_BELANJA")["REALISASI"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# --------------------------------------------------------------------------
# Proyeksi: rerata tertimbang tingkat realisasi 5 tahun sebelumnya x pagu tahun berjalan
# (lihat common.py::hitung_proyeksi_agregat utk rumus lengkap & penjelasan bobot)
# --------------------------------------------------------------------------

FILTER_ENTITAS = {"KDDEPT": kddept, "KDSATKER": kdsatker}

proyeksi_agregat_bulanan, tahun_dipakai = hitung_proyeksi_agregat(df, tahun, pagu_total, FILTER_ENTITAS)

if proyeksi_agregat_bulanan is None:
    # Tidak ada histori sama sekali (mis. tahun pertama dalam data) -> fallback rata-rata bulan berjalan
    rerata_bulanan = (kumulatif.iloc[bulan_terakhir - 1] / bulan_terakhir) if bulan_terakhir else 0
    proyeksi_akhir_tahun = rerata_bulanan * 12
    metode_proyeksi = "fallback"
else:
    # Target satu tahun penuh berdasarkan rerata tertimbang tingkat realisasi historis.
    # Proyeksi akhir tahun = target itu, ATAU realisasi aktual sekarang kalau realisasi
    # aktual sudah melebihi target itu sendiri (tidak mungkin proyeksi akhir tahun lebih
    # kecil dari yang sudah benar-benar terealisasi).
    target_tahun_penuh = proyeksi_agregat_bulanan.sum()
    proyeksi_akhir_tahun = max(realisasi_total, target_tahun_penuh)
    metode_proyeksi = "historis"

persen_proyeksi = (proyeksi_akhir_tahun / pagu_total * 100) if pagu_total else 0


# --------------------------------------------------------------------------
# Header & KPI
# --------------------------------------------------------------------------

st.title("📊 Dashboard Pagu & Realisasi Satker")
st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data()}")
st.caption(f"{nmdept} — {nmsatker} — Tahun {tahun}")

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    kpi_card("Pagu", f"Rp {pagu_total:,.0f}")
with r1c2:
    kpi_card("Realisasi", f"Rp {realisasi_total:,.0f}", f"{persen_serapan:.1f}% dari pagu")
with r2c1:
    kpi_card("Sisa Pagu", f"Rp {sisa_pagu:,.0f}")
with r2c2:
    kpi_card(
        "Proyeksi Realisasi Akhir Tahun",
        f"Rp {proyeksi_akhir_tahun:,.0f}",
        f"{persen_proyeksi:.1f}% dari pagu",
    )

st.divider()

# --------------------------------------------------------------------------
# Grafik batang: Pagu vs Realisasi per bulan (kumulatif) + Grafik batang total
# --------------------------------------------------------------------------

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Realisasi per Bulan")
    bar_df = pd.DataFrame({"Bulan": BULAN_KOLOM, "Realisasi": monthly.values})
    fig_bar = px.bar(bar_df, x="Bulan", y="Realisasi", text_auto=".2s")
    fig_bar.update_layout(yaxis_title="Rupiah", xaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.subheader("Pagu vs Realisasi")
    fig_pv = go.Figure(data=[
        go.Bar(name="Pagu", x=["Total"], y=[pagu_total]),
        go.Bar(name="Realisasi", x=["Total"], y=[realisasi_total]),
    ])
    fig_pv.update_layout(barmode="group", yaxis_title="Rupiah")
    st.plotly_chart(fig_pv, use_container_width=True)


# --------------------------------------------------------------------------
# Pie charts
# --------------------------------------------------------------------------

st.subheader("Komposisi")
p1, p2 = st.columns(2)

with p1:
    st.caption("Realisasi vs Sisa Pagu")
    fig_pie1 = px.pie(
        names=["Realisasi", "Sisa Pagu"],
        values=[realisasi_total, max(sisa_pagu, 0)],
        hole=0.4,
    )
    st.plotly_chart(fig_pie1, use_container_width=True)

with p2:
    st.caption("Realisasi per Jenis Belanja")
    fig_pie2 = px.pie(jenis_belanja, names="LABEL_JENIS_BELANJA", values="REALISASI", hole=0.4)
    st.plotly_chart(fig_pie2, use_container_width=True)


# --------------------------------------------------------------------------
# Trendline proyeksi
# --------------------------------------------------------------------------

st.subheader("Tren & Proyeksi Realisasi hingga Akhir Tahun")

bulan_angka = list(range(1, 13))


# Grafik non-kumulatif: nilai realisasi per bulan (biar kelihatan naik-turunnya). Bulan yang
# belum benar-benar berakhir (lihat bulan_penuh_terakhir) ditampilkan sebagai proyeksi (garis
# putus-putus) memakai nilai proyeksi akhir bulan, BUKAN realisasi parsial yang sudah tercatat
# sejauh ini -- walaupun datanya sudah tidak nol.
def _nilai_proyeksi_bulan(b):
    if proyeksi_agregat_bulanan is not None:
        return proyeksi_agregat_bulanan[b - 1]
    return rerata_bulanan


aktual = [monthly.values[b - 1] if b <= bulan_penuh_terakhir else None for b in bulan_angka]
proyeksi = []
for b in bulan_angka:
    if bulan_penuh_terakhir == 0:
        # Belum ada satu bulan pun yang penuh datanya tahun ini -> seluruh garis adalah proyeksi
        proyeksi.append(_nilai_proyeksi_bulan(b))
    elif b < bulan_penuh_terakhir:
        proyeksi.append(None)
    elif b == bulan_penuh_terakhir:
        proyeksi.append(monthly.values[b - 1])  # titik sambung dengan garis aktual
    else:
        proyeksi.append(_nilai_proyeksi_bulan(b))

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=BULAN_KOLOM, y=aktual, mode="lines+markers",
    name="Realisasi per Bulan (Aktual)",
    line=dict(width=3, shape="spline", smoothing=1.1),
))
fig_trend.add_trace(go.Scatter(
    x=BULAN_KOLOM, y=proyeksi, mode="lines+markers",
    name="Proyeksi (rata-rata bulanan)",
    line=dict(dash="dash", shape="spline", smoothing=1.1),
))
fig_trend.update_layout(yaxis_title="Rupiah (per bulan)", xaxis_title=None)
st.plotly_chart(fig_trend, use_container_width=True)

_catatan_bulan_berjalan = ""
if bulan_terakhir > bulan_penuh_terakhir:
    _catatan_bulan_berjalan = (
        f" Bulan {BULAN_LABEL.get(bulan_terakhir, '-')} sendiri masih berjalan (belum berakhir), "
        "jadi titik & garisnya di grafik ini memakai proyeksi akhir bulan, bukan realisasi yang "
        "baru tercatat sebagian sejauh ini."
    )
_label_batas = BULAN_LABEL.get(bulan_penuh_terakhir, "-") if bulan_penuh_terakhir else None

if metode_proyeksi == "historis":
    daftar_tahun_ket = ", ".join(str(t) for t in tahun_dipakai)
    _batas_teks = f"Bulan setelah {_label_batas}" if _label_batas else "Seluruh bulan tahun ini"
    st.caption(
        "Grafik ini menampilkan realisasi tiap bulan (bukan kumulatif) supaya terlihat bulan mana "
        f"yang realisasinya naik/turun. {_batas_teks} adalah proyeksi yang dihitung dari rerata "
        "tertimbang tingkat realisasi tahun-tahun sebelumnya (bobot 50%-25%-12,5%-6,25%-6,25% "
        f"untuk tahun y-1 s.d. y-5) dikalikan pagu tahun {tahun}. Tahun historis yang tersedia & "
        f"dipakai: {daftar_tahun_ket}.{_catatan_bulan_berjalan}"
    )
else:
    _batas_teks = f"Bulan setelah {_label_batas}" if _label_batas else "Seluruh bulan tahun ini"
    st.caption(
        "Grafik ini menampilkan realisasi tiap bulan (bukan kumulatif) supaya terlihat bulan mana "
        f"yang realisasinya naik/turun. {_batas_teks} adalah proyeksi. Belum ada data historis "
        f"(tahun sebelum {tahun}) untuk entitas ini, sehingga proyeksi memakai metode cadangan: "
        f"rata-rata realisasi per bulan pada tahun berjalan dikalikan 12 bulan.{_catatan_bulan_berjalan}"
    )

# --------------------------------------------------------------------------
# Tabel realisasi per bulan per jenis belanja (aktual vs proyeksi), ditranspose:
# kolom = jenis belanja, baris = bulan (+ baris ringkasan total di bawah).
# --------------------------------------------------------------------------

st.markdown("**Realisasi Bulanan per Jenis Belanja**")

# Tanggal "sd." di judul baris total realisasi disamakan dengan tanggal update file data
# (lihat tanggal_update_data() di common.py) -- ambil bagian tanggalnya saja, tanpa jam.
_tgl_update_lengkap = tanggal_update_data()
_tgl_saja = _tgl_update_lengkap.split(",")[0].strip() if "," in _tgl_update_lengkap else _tgl_update_lengkap

BARIS_TOTAL_REALISASI_RP = f"Total Realisasi (Rp) sd. tanggal {_tgl_saja}"
BARIS_TOTAL_REALISASI_PCT = f"Total Realisasi (%) sd. tanggal {_tgl_saja}"
BARIS_SISA_PAGU = "Sisa Pagu"
BARIS_BLOKIR = "Blokir"
BARIS_TOTAL_PROYEKSI_RP = "Total Realisasi + Proyeksi Akhir Tahun (Rp)"
BARIS_TOTAL_PROYEKSI_PCT = "Total Realisasi + Proyeksi Akhir Tahun (%)"

# Urutan kolom tabel mengikuti kode jenis belanja (51 Pegawai, 52 Barang, 53 Modal, dst),
# bukan diurutkan berdasarkan besar realisasi seperti pie chart.
urutan_kode = (
    df_satker[["JENIS BELANJA", "LABEL_JENIS_BELANJA"]]
    .drop_duplicates()
    .sort_values("JENIS BELANJA")["LABEL_JENIS_BELANJA"]
    .tolist()
)

pagu_per_jenis = (
    df_satker.groupby("LABEL_JENIS_BELANJA")["PAGU"].sum()
    .reindex(urutan_kode)
)

# Data realisasi AKTUAL murni (sebelum bulan yang belum penuh ditimpa angka proyeksi) --
# dipakai untuk baris "Total Realisasi" (Rp & %), yang HANYA menghitung uang yang sudah
# benar-benar terealisasi (tidak termasuk proyeksi bulan yang belum berakhir).
realisasi_aktual_jenis = (
    df_satker.groupby("LABEL_JENIS_BELANJA")[BULAN_KOLOM].sum()
    .astype(float)
    .reindex(urutan_kode)
)

proyeksi_per_jenis, _ = hitung_proyeksi_per_kategori(
    df, tahun, pagu_per_jenis.reindex(urutan_kode), FILTER_ENTITAS, "LABEL_JENIS_BELANJA"
)

# Data tampilan per bulan: realisasi aktual utk bulan yang sudah penuh, proyeksi utk bulan
# yang belum berakhir/belum terjadi (memakai bulan_penuh_terakhir, sama seperti grafik tren).
# Belanja Pegawai (51) pakai rumus proyeksi berbeda & TIDAK dibatasi maks pagu (lihat
# common.py::_hitung_proyeksi_belanja_pegawai); jenis belanja lain otomatis dibatasi maks
# 100% dari pagu -- lihat common.py::isi_tabel_proyeksi utk penjelasan lengkap cara hitungnya.
tabel_tampil = isi_tabel_proyeksi(
    realisasi_aktual_jenis, proyeksi_per_jenis, pagu_per_jenis, bulan_penuh_terakhir,
    kategori_dikecualikan_cap=LABEL_BELANJA_PEGAWAI,
)

# --- Transpose: baris = bulan, kolom = jenis belanja, + kolom TOTAL di kanan ---
tabel_t = tabel_tampil.reindex(urutan_kode).T
tabel_t["TOTAL"] = tabel_t.sum(axis=1)

pagu_row = pagu_per_jenis.reindex(urutan_kode).copy()
pagu_row["TOTAL"] = pagu_total
pagu_row.name = "PAGU"

pagu_aman = pagu_per_jenis.reindex(urutan_kode).replace(0, np.nan)  # hindari bagi nol

total_real_rp = realisasi_aktual_jenis.sum(axis=1)
total_real_rp["TOTAL"] = total_real_rp.sum()
total_real_rp.name = BARIS_TOTAL_REALISASI_RP

total_real_pct = (realisasi_aktual_jenis.sum(axis=1) / pagu_aman * 100).fillna(0)
total_real_pct["TOTAL"] = (total_real_rp["TOTAL"] / pagu_total * 100) if pagu_total else 0
total_real_pct.name = BARIS_TOTAL_REALISASI_PCT

# Sisa Pagu = Pagu - realisasi AKTUAL (bukan realisasi+proyeksi), per jenis belanja.
sisa_pagu_row = (pagu_per_jenis.reindex(urutan_kode) - realisasi_aktual_jenis.sum(axis=1))
sisa_pagu_row["TOTAL"] = pagu_total - total_real_rp["TOTAL"]
sisa_pagu_row.name = BARIS_SISA_PAGU

blokir_row = df_satker.groupby("LABEL_JENIS_BELANJA")["BLOKIR"].sum().reindex(urutan_kode).fillna(0)
blokir_row["TOTAL"] = df_satker["BLOKIR"].sum()
blokir_row.name = BARIS_BLOKIR

total_proyeksi_rp = tabel_tampil.reindex(urutan_kode).sum(axis=1)
total_proyeksi_rp["TOTAL"] = total_proyeksi_rp.sum()
total_proyeksi_rp.name = BARIS_TOTAL_PROYEKSI_RP

total_proyeksi_pct = (tabel_tampil.reindex(urutan_kode).sum(axis=1) / pagu_aman * 100).fillna(0)
total_proyeksi_pct["TOTAL"] = (total_proyeksi_rp["TOTAL"] / pagu_total * 100) if pagu_total else 0
total_proyeksi_pct.name = BARIS_TOTAL_PROYEKSI_PCT

tabel_final = pd.concat([
    pagu_row.to_frame().T,
    tabel_t,
    total_real_rp.to_frame().T,
    total_real_pct.to_frame().T,
    sisa_pagu_row.to_frame().T,
    blokir_row.to_frame().T,
    total_proyeksi_rp.to_frame().T,
    total_proyeksi_pct.to_frame().T,
])
tabel_final = tabel_final.reindex(columns=urutan_kode + ["TOTAL"])

BARIS_RUPIAH = ["PAGU"] + BULAN_KOLOM + [
    BARIS_TOTAL_REALISASI_RP, BARIS_SISA_PAGU, BARIS_BLOKIR, BARIS_TOTAL_PROYEKSI_RP,
]
BARIS_PERSEN = [BARIS_TOTAL_REALISASI_PCT, BARIS_TOTAL_PROYEKSI_PCT]

# Baris bulan yang proyeksi (belum berakhir) ditandai kuning; begitu juga baris ringkasan
# "Total Realisasi + Proyeksi" karena mengandung angka proyeksi (kalau memang ada proyeksinya).
baris_bulan_proyeksi = [b for i, b in enumerate(BULAN_KOLOM) if i >= bulan_penuh_terakhir]
mask_final = pd.DataFrame(False, index=tabel_final.index, columns=tabel_final.columns)
mask_final.loc[baris_bulan_proyeksi, :] = True
if bulan_penuh_terakhir < 12:
    mask_final.loc[BARIS_TOTAL_PROYEKSI_RP, :] = True
    mask_final.loc[BARIS_TOTAL_PROYEKSI_PCT, :] = True


def _style_tabel(_):
    styles = pd.DataFrame(
        np.where(mask_final, "background-color: #fff3cd; color: #7a5b00;", ""),
        index=mask_final.index, columns=mask_final.columns,
    )
    # Baris PAGU dicetak tebal supaya jelas beda dari baris realisasi bulanan
    styles.loc["PAGU", :] = styles.loc["PAGU", :] + "font-weight: bold;"
    return styles


styled_tabel = (
    tabel_final.style
    .apply(_style_tabel, axis=None)
    .format("Rp {:,.0f}", subset=pd.IndexSlice[BARIS_RUPIAH, :])
    .format("{:.1f}%", subset=pd.IndexSlice[BARIS_PERSEN, :])
)
st.dataframe(styled_tabel, use_container_width=True)
st.caption(
    "🟨 Sel berwarna kuning = mengandung angka proyeksi (bulan yang belum berakhir), dihitung "
    "dari rerata tertimbang tingkat realisasi jenis belanja ini pada tahun-tahun sebelumnya "
    "(lihat penjelasan di atas grafik tren) dikalikan pagu jenis belanja tahun berjalan. Khusus "
    f"**{LABEL_BELANJA_PEGAWAI}**, proyeksi dihitung dari rerata tertimbang REALISASI bulanan "
    "tahun-tahun sebelumnya secara langsung (bobot 50%-25%-12,5%-6,25%-6,25% untuk y-1 s.d. "
    f"y-5), TANPA dibatasi maksimal pagu tahun berjalan -- karena realisasi {LABEL_BELANJA_PEGAWAI} "
    "bisa saja melebihi pagu saat ini. Jika suatu jenis belanja belum punya histori, dipakai "
    "rata-rata realisasi tahun berjalan sebagai cadangan. Baris \"Total Realisasi\" hanya "
    "menjumlahkan uang yang sudah benar-benar terealisasi (bulan penuh saja), sedangkan baris "
    "\"Total Realisasi + Proyeksi Akhir Tahun\" menjumlahkan realisasi ditambah estimasi "
    "bulan-bulan yang belum berakhir/belum terjadi. Kolom PAGU & baris PAGU tidak ditandai "
    "kuning karena berupa acuan, bukan proyeksi."
)

st.divider()


# --------------------------------------------------------------------------
# Pencarian tematik lintas satker/kementerian/provinsi -- dipakai AI di chat box
# untuk menjawab pertanyaan seperti "berapa pagu ketahanan pangan di Riau?" atau
# "penanganan karhutla ada di satker mana saja?".
# --------------------------------------------------------------------------

def cari_anggaran(kata_kunci: list, provinsi: str = None, tahun_cari: int = None) -> dict:
    d = df
    if SCOPE_KDSATKER is not None:
        # User satker biasa: pencarian dibatasi ke data satker miliknya sendiri saja,
        # tidak boleh melihat/menghitung data satker lain.
        d = d[d["KDSATKER"] == SCOPE_KDSATKER]
    d = d[d["TAHUN"] == (tahun_cari or tahun)]

    if provinsi:
        if "PROVINSI" in d.columns:
            d = d[d["PROVINSI"].str.contains(provinsi, case=False, na=False)]
        else:
            return {"error": "Kolom provinsi tidak tersedia di data ini."}

    if kata_kunci:
        mask = pd.Series(False, index=d.index)
        for kw in kata_kunci:
            mask = mask | d["_TEKS_CARI"].str.contains(str(kw).lower(), na=False)
        d = d[mask]

    if d.empty:
        return {
            "ditemukan": False,
            "pesan": (
                "Tidak ada baris data yang cocok dengan kata kunci/filter ini. Kemungkinan "
                "temanya tidak tercatat secara eksplisit di nama program/kegiatan/output pada "
                "level detail yang tersedia di data ini."
            ),
        }

    rincian = (
        d.groupby(["KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER"])
        .agg(PAGU=("PAGU", "sum"), REALISASI=("REALISASI", "sum"))
        .reset_index()
        .sort_values("PAGU", ascending=False)
        .head(30)
    )
    rincian["KDSATKER"] = rincian["KDSATKER"].apply(fmt_satker)
    rincian["KDDEPT"] = rincian["KDDEPT"].apply(fmt_dept)

    return {
        "ditemukan": True,
        "tahun": int(tahun_cari or tahun),
        "jumlah_baris_cocok": int(len(d)),
        "jumlah_satker_ditemukan": int(rincian.shape[0]),
        "total_pagu": float(d["PAGU"].sum()),
        "total_realisasi": float(d["REALISASI"].sum()),
        "rincian_per_satker_top30": rincian.to_dict(orient="records"),
        "catatan": (
            "rincian_per_satker_top30 diurutkan dari pagu terbesar, dibatasi 30 baris teratas. "
            "total_pagu & total_realisasi sudah menjumlahkan SEMUA satker yang cocok, tidak "
            "hanya yang ditampilkan di rincian."
            + (
                " Pencarian ini dibatasi hanya pada data satker Anda sendiri (bukan lintas satker)."
                if SCOPE_KDSATKER is not None else ""
            )
        ),
    }


def ringkasan_data_untuk_ai() -> str:
    top3_jenis = jenis_belanja.head(3)
    baris_jenis = "\n".join(
        f"- {row.LABEL_JENIS_BELANJA}: Rp {row.REALISASI:,.0f}"
        for row in top3_jenis.itertuples()
    )
    kddept_ket = f" (kode {fmt_dept(kddept)})" if kddept is not None else ""
    kdsatker_ket = f" (kode {fmt_satker(kdsatker)})" if kdsatker is not None else ""
    return f"""
Data satker:
- Kementerian/Lembaga: {nmdept}{kddept_ket}
- Satker: {nmsatker}{kdsatker_ket}
- Tahun: {tahun}
- Pagu: Rp {pagu_total:,.0f}
- Realisasi sampai bulan {BULAN_LABEL.get(bulan_terakhir, '-')}: Rp {realisasi_total:,.0f} ({persen_serapan:.1f}% dari pagu)
- Sisa pagu: Rp {sisa_pagu:,.0f}
- Proyeksi realisasi akhir tahun: Rp {proyeksi_akhir_tahun:,.0f} ({persen_proyeksi:.1f}% dari pagu)
- 3 jenis belanja dengan realisasi terbesar:
{baris_jenis}
""".strip()


# --------------------------------------------------------------------------
# AI: narasi otomatis + chat pencarian tematik
# --------------------------------------------------------------------------

_deskripsi_tool = (
    (
        "Mencari & menjumlahkan pagu/realisasi anggaran di SELURUH data (semua "
        "kementerian & satker, bukan cuma yang sedang dipilih di dashboard), "
        if SCOPE_KDSATKER is None else
        "Mencari & menjumlahkan pagu/realisasi anggaran DI DALAM DATA SATKER INI SAJA "
        "(seluruh tahun & tema yang tersedia untuk satker ini, bukan cuma yang sedang "
        "dipilih di dashboard; TIDAK bisa mengakses data satker lain), "
    )
    + "berdasarkan kata kunci tema/program/kegiatan/output, dan opsional filter "
    "provinsi atau tahun. WAJIB dipakai untuk pertanyaan yang menyebutkan tema "
    "(mis. 'ketahanan pangan', 'kebakaran hutan'), lokasi/provinsi tertentu, atau "
    "kementerian/satker yang BUKAN yang sedang aktif di dashboard."
)

render_ai_section(
    ringkasan_data_untuk_ai, cari_anggaran, page_key="satker",
    narasi_variasi_key=f"{tahun}_{kddept}_{kdsatker}",
    deskripsi_tool=_deskripsi_tool,
)

if is_super:
    st.divider()
    render_dataset_upload_qa(page_key="upload_satker")
