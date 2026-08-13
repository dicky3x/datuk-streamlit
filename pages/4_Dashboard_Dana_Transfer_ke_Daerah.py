"""
Halaman 4: Dashboard Dana Transfer ke Daerah (TKD)
-----------------------------------------------------
Memakai sumber data yang SAMA dengan Halaman 1 (data/pagu_realisasi.csv.gz), difilter
hanya ke kategori Transfer ke Daerah (kode jenis belanja 61-66: DBH, DAU, DAK Fisik,
DAK Nonfisik, Insentif Fiskal, Dana Desa).

Akses: KHUSUS super user (TKD disalurkan KPPN ke kabupaten/kota, bukan per satker K/L,
jadi tidak relevan dibatasi ke satker individual). Halaman ini otomatis tidak muncul di
navigasi untuk user satker biasa -- lihat app.py.

Catatan: grafik tren & tabel bulanan proyeksi sengaja TIDAK ada di halaman ini (hanya
ada di Halaman 1) sesuai permintaan; KPI "Proyeksi Realisasi Akhir Tahun" tetap ada.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    BULAN_KOLOM, get_data, tanggal_update_data, kpi_card,
    hitung_proyeksi_agregat, hitung_bulan_penuh_terakhir, KODE_JENIS_TKD,
    buat_cari_generik, render_ai_section,
)
import numpy as np

df_semua = get_data()

# Login sudah ditangani app.py (router) -- halaman ini pun cuma dimasukkan ke navigasi
# kalau super user (lihat app.py), tapi kita cek ulang di sini sbg proteksi tambahan
# (jaga-jaga kalau ada yang coba akses langsung via URL).
auth = st.session_state.auth
if auth["role"] != "super":
    st.error("Halaman ini hanya bisa diakses oleh super user.")
    st.stop()

if "KABKOTA" not in df_semua.columns:
    st.error(
        "Kolom KABKOTA tidak ditemukan di data. Halaman ini butuh kolom kabupaten/kota "
        "untuk membagi tampilan Dana Transfer ke Daerah."
    )
    st.stop()

df = df_semua[df_semua["JENIS BELANJA"].isin(KODE_JENIS_TKD)]

if df.empty:
    st.warning("Tidak ada data Transfer ke Daerah (kode jenis belanja 61-66) di dataset ini.")
    st.stop()


# --------------------------------------------------------------------------
# Sidebar - filter
# --------------------------------------------------------------------------

st.sidebar.header("Filter")

tahun_list = sorted(df["TAHUN"].unique(), reverse=True)
tahun = st.sidebar.selectbox("Tahun", tahun_list)
df_tahun = df[df["TAHUN"] == tahun]

SEMUA_KABKOTA = "— Semua Kabupaten/Kota —"
kabkota_list = sorted(df_tahun["KABKOTA"].dropna().unique().tolist())
kabkota_pilih = st.sidebar.selectbox("Kabupaten/Kota", [SEMUA_KABKOTA] + kabkota_list)

if kabkota_pilih == SEMUA_KABKOTA:
    kabkota = None
    df_wilayah = df_tahun
else:
    kabkota = kabkota_pilih
    df_wilayah = df_tahun[df_tahun["KABKOTA"] == kabkota]


# --------------------------------------------------------------------------
# Agregasi
# --------------------------------------------------------------------------

pagu_total = df_wilayah["PAGU"].sum()
realisasi_total = df_wilayah["REALISASI"].sum()
sisa_pagu = df_wilayah["SISA PAGU"].sum()
persen_serapan = (realisasi_total / pagu_total * 100) if pagu_total else 0

monthly = df_wilayah[BULAN_KOLOM].sum()
kumulatif = monthly.cumsum()

bulan_terakhir, bulan_penuh_terakhir = hitung_bulan_penuh_terakhir(df_wilayah, tahun)

jenis_tkd = (
    df_wilayah.groupby("LABEL_JENIS_BELANJA")["REALISASI"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)


# --------------------------------------------------------------------------
# Proyeksi (dipakai KPI card saja -- lihat catatan di atas soal grafik tren/tabel)
# --------------------------------------------------------------------------

FILTER_ENTITAS = {"KABKOTA": kabkota}

proyeksi_agregat_bulanan, tahun_dipakai = hitung_proyeksi_agregat(df, tahun, pagu_total, FILTER_ENTITAS)

if proyeksi_agregat_bulanan is None:
    rerata_bulanan = (kumulatif.iloc[bulan_terakhir - 1] / bulan_terakhir) if bulan_terakhir else 0
    proyeksi_akhir_tahun = rerata_bulanan * 12
else:
    target_tahun_penuh = proyeksi_agregat_bulanan.sum()
    proyeksi_akhir_tahun = max(realisasi_total, target_tahun_penuh)

persen_proyeksi = (proyeksi_akhir_tahun / pagu_total * 100) if pagu_total else 0


# --------------------------------------------------------------------------
# Header & KPI
# --------------------------------------------------------------------------

st.title("🏘️ Dashboard Dana Transfer ke Daerah (TKD)")
st.caption(f"🕒 Data terakhir diperbarui: {tanggal_update_data()}")
st.caption(f"{kabkota or 'Semua Kabupaten/Kota'} — Tahun {tahun}")

r1c1, r1c2 = st.columns(2)
r2c1, r2c2 = st.columns(2)

with r1c1:
    kpi_card("Pagu TKD", f"Rp {pagu_total:,.0f}")
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
# Grafik batang
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
    st.caption("Realisasi per Jenis TKD")
    fig_pie2 = px.pie(jenis_tkd, names="LABEL_JENIS_BELANJA", values="REALISASI", hole=0.4)
    st.plotly_chart(fig_pie2, use_container_width=True)


# --------------------------------------------------------------------------
# Perbandingan antar kabupaten/kota (hanya relevan kalau "Semua" yang dipilih) --
# ATAU rincian per jenis belanja TKD (kalau satu kabupaten/kota spesifik dipilih)
# --------------------------------------------------------------------------

if kabkota is None:
    st.subheader("Perbandingan antar Kabupaten/Kota")
    per_kabkota = (
        df_tahun.groupby("KABKOTA")
        .agg(Pagu=("PAGU", "sum"), Realisasi=("REALISASI", "sum"))
        .reset_index()
        .sort_values("Pagu", ascending=False)
    )
    per_kabkota["Persen Realisasi"] = (
        per_kabkota["Realisasi"] / per_kabkota["Pagu"].replace(0, np.nan) * 100
    ).fillna(0)

    fig_kabkota = go.Figure(data=[
        go.Bar(name="Pagu", x=per_kabkota["KABKOTA"], y=per_kabkota["Pagu"]),
        go.Bar(name="Realisasi", x=per_kabkota["KABKOTA"], y=per_kabkota["Realisasi"]),
    ])
    fig_kabkota.update_layout(barmode="group", yaxis_title="Rupiah", xaxis_title=None)
    st.plotly_chart(fig_kabkota, use_container_width=True)

    st.dataframe(
        per_kabkota.style.format({
            "Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}", "Persen Realisasi": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.subheader(f"Rincian per Jenis Belanja — {kabkota}")
    per_jenis = (
        df_wilayah.groupby("LABEL_JENIS_BELANJA")
        .agg(Pagu=("PAGU", "sum"), Realisasi=("REALISASI", "sum"), **{"Sisa Pagu": ("SISA PAGU", "sum")})
        .reset_index()
        .rename(columns={"LABEL_JENIS_BELANJA": "Jenis Belanja"})
        .sort_values("Pagu", ascending=False)
    )
    per_jenis["% Realisasi"] = (
        per_jenis["Realisasi"] / per_jenis["Pagu"].replace(0, np.nan) * 100
    ).fillna(0)

    baris_total = pd.DataFrame([{
        "Jenis Belanja": "TOTAL",
        "Pagu": per_jenis["Pagu"].sum(),
        "Realisasi": per_jenis["Realisasi"].sum(),
        "Sisa Pagu": per_jenis["Sisa Pagu"].sum(),
        "% Realisasi": (per_jenis["Realisasi"].sum() / per_jenis["Pagu"].sum() * 100) if per_jenis["Pagu"].sum() else 0,
    }])
    per_jenis_tampil = pd.concat([per_jenis, baris_total], ignore_index=True)
    per_jenis_tampil = per_jenis_tampil[["Jenis Belanja", "Pagu", "Realisasi", "% Realisasi", "Sisa Pagu"]]

    def _tebalkan_total(row):
        return ["font-weight: bold;" if row["Jenis Belanja"] == "TOTAL" else "" for _ in row]

    st.dataframe(
        per_jenis_tampil.style
        .apply(_tebalkan_total, axis=1)
        .format({
            "Pagu": "Rp {:,.0f}", "Realisasi": "Rp {:,.0f}",
            "% Realisasi": "{:.1f}%", "Sisa Pagu": "Rp {:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()


# --------------------------------------------------------------------------
# AI: narasi otomatis + chat pencarian tematik
# --------------------------------------------------------------------------

def _ringkasan():
    return f"""
Data Transfer ke Daerah (TKD):
- Wilayah: {kabkota or 'Semua Kabupaten/Kota'}
- Tahun: {tahun}
- Pagu: Rp {pagu_total:,.0f}
- Realisasi: Rp {realisasi_total:,.0f} ({persen_serapan:.1f}% dari pagu)
- Sisa pagu: Rp {sisa_pagu:,.0f}
- Proyeksi realisasi akhir tahun: Rp {proyeksi_akhir_tahun:,.0f} ({persen_proyeksi:.1f}% dari pagu)
""".strip()


cari_fn = buat_cari_generik(
    df, ["NMDEPT", "NMSATKER", "KABKOTA", "LABEL_JENIS_BELANJA"], scope_kdsatker=None,
)

render_ai_section(
    _ringkasan, cari_fn, page_key="tkd",
    narasi_variasi_key=f"{tahun}_{kabkota}",
    deskripsi_tool=(
        "Mencari & menjumlahkan pagu/realisasi Transfer ke Daerah berdasarkan kata kunci "
        "(nama satker/kabupaten-kota/jenis TKD), dan opsional filter provinsi atau tahun."
    ),
)
