"""
forecasting.py
----------------
Mesin proyeksi hybrid untuk Halaman 1 (Dashboard Pagu & Realisasi Satker).

Menggantikan pendekatan lama (rerata tertimbang tingkat realisasi x pagu tahun berjalan,
lihat common.py::hitung_proyeksi_agregat) dengan pendekatan yang lebih detail & adaptif:

1.  PROFIL TAHUNAN & BULANAN per kombinasi (satker, akun)
    Untuk tiap kombinasi satker-akun, dihitung dua besaran dari histori 5 tahun terakhir,
    keduanya dirata-ratakan tertimbang dengan bobot 2025=50%, 2024=25%, 2023=12,5%,
    2022=6,25%, 2021=6,25% (lihat BOBOT_TAHUN):
      - target tahunan: tingkat realisasi (realisasi/pagu) tertimbang x pagu tahun berjalan
        (untuk Belanja Pegawai/51: rerata tertimbang REALISASI rupiah langsung, TIDAK
        diskalakan ke pagu -- lihat penjelasan di _hitung_target_tahunan)
      - profil bulanan: proporsi tiap bulan terhadap total realisasi tahun ybs, dirata-ratakan
        tertimbang lalu dinormalisasi ulang supaya totalnya tetap 100%

2.  ROLLING FORECAST
    Begitu realisasi bulan terbaru tersedia, forecast bulan-bulan berikutnya dihitung ULANG:
    bulan yang sudah terealisasi dipakai apa adanya (aktual), dan SISA target tahunan
    (target tahunan - realisasi sd sekarang) didistribusikan ke bulan-bulan tersisa memakai
    profil bulanan historis yang sudah dinormalisasi ulang HANYA untuk bulan-bulan tersisa itu.
    Artinya proporsi historis bulan-bulan yang sudah lewat tidak lagi "dihitung" di ulang.

3.  FALLBACK BERJENJANG untuk kombinasi satker-akun tanpa histori (mis. satker baru):
    (a) profil sendiri (kalau ada) -> (b) profil "sejenis": rata-rata satker lain dengan
    kementerian, jenis belanja, & kelompok pagu yang sama -> (c) rata-rata seluruh satker
    untuk akun/jenis belanja yang sama.

4.  PENYESUAIAN KEBIJAKAN (khusus Belanja Pegawai) lewat tabel konfigurasi KONFIGURASI_KEBIJAKAN
    -- lihat fungsi _terapkan_kebijakan. Menambah aturan baru = menambah 1 dict baru di
    konfigurasi, TANPA perlu mengubah logika kode.

Semua fungsi berat (agregasi 5 tahun histori) ditulis vektor (groupby+numpy), bukan loop
per baris, supaya tetap cepat walau datanya ratusan ribu baris.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from common import (
    BULAN_KOLOM, BOBOT_TAHUN, LABEL_JENIS_BELANJA_SINGKAT, LABEL_BELANJA_PEGAWAI,
)

# --------------------------------------------------------------------------
# 1. Konfigurasi kebijakan (policy adjustment layer)
# --------------------------------------------------------------------------
# Tiap aturan berlaku untuk kombinasi (kddept, pola nama akun) di tahun tertentu, mengubah
# "indeks" pembayaran suatu komponen Belanja Pegawai mulai bulan tertentu, dengan opsi rapel
# (pembayaran rapelan/susulan selisih bulan-bulan sebelum rapel diproses) di bulan tertentu.
#
# Field:
#   nama            -- label deskriptif (ditampilkan di UI/caption)
#   kddept          -- kode kementerian/lembaga (int) yang terkena aturan ini
#   akun_pola       -- substring (case-insensitive) untuk mencocokkan kolom AKUN
#   tahun_efektif   -- tahun anggaran berlakunya aturan ini
#   indeks_lama     -- indeks/rate yang berlaku sebelum kenaikan (mis. 0.70 = 70%)
#   indeks_baru     -- indeks/rate baru mulai bulan_mulai
#   bulan_mulai     -- bulan (1-12) mulai berlakunya indeks baru SECARA SUBSTANSI
#   bulan_rapel     -- bulan (1-12) saat kenaikan baru benar-benar DIBAYARKAN beserta
#                      rapel (susulan) untuk bulan_mulai..bulan_rapel-1. Kalau bulan_rapel
#                      sama dengan bulan_mulai, tidak ada rapel (indeks baru langsung berlaku
#                      penuh sejak bulan_mulai).
KONFIGURASI_KEBIJAKAN = [
    {
        "nama": "Kenaikan Tunjangan Kinerja Kementerian Pertahanan (akun 512411, +28,57% mulai September, tanpa rapel Juli-Agustus)",
        "kddept": 12,
        "akun_pola": "tunjangan khusus/kegiatan/kinerja",
        "tahun_efektif": 2026,
        "indeks_lama": 0.70,
        "indeks_baru": 0.90,
        "bulan_mulai": 9,
        "bulan_rapel": 9,
    },
]


# --------------------------------------------------------------------------
# 1b. Konfigurasi ANOMALI SATU KALI (one-off, tidak berulang)
# --------------------------------------------------------------------------
# Berbeda dengan KONFIGURASI_KEBIJAKAN (menyesuaikan bulan-bulan MASA DEPAN yang masih
# diproyeksikan), tabel ini menandai bulan-bulan yang SUDAH TERJADI di tahun berjalan sebagai
# "anomali tidak berulang" -- misalnya rapel/pembayaran susulan yang terjadi sekali karena
# keterlambatan administratif, bukan pola yang akan berulang tiap tahun. Bulan yang ditandai
# di sini TETAP ditampilkan apa adanya di angka aktual/realisasi (uangnya memang sudah keluar),
# tapi DIKECUALIKAN saat menghitung "kecepatan belanja rata-rata" (dipakai utk override
# target tahunan & deteksi lonjakan tidak wajar) -- supaya satu bulan yang njomplang tidak
# membuat proyeksi bulan-bulan normal berikutnya ikut meroket.
#
# Field: nama, kddept (None = semua), jenis_belanja (None = semua), tahun, bulan (1-12).
PENYESUAIAN_SATU_KALI = [
    {
        "nama": (
            "Rapel gaji Okt-Des 2025 yang terlambat, dibayarkan April 2026 "
            "(satker lingkup BA Mahkamah Agung)"
        ),
        "kddept": 5,  # Mahkamah Agung
        "jenis_belanja": 51,
        "tahun": 2026,
        "bulan": 4,
    },
]


def _masker_anomali_satu_kali(kddept_arr: np.ndarray, jenis_arr: np.ndarray, tahun_y: int) -> np.ndarray:
    """Bangun matriks boolean (n_baris x 12) yang menandai bulan mana per baris yang harus
    dikecualikan dari perhitungan kecepatan belanja (bukan dari angka aktual itu sendiri)."""
    n = len(kddept_arr)
    mask = np.zeros((n, 12), dtype=bool)
    for aturan in PENYESUAIAN_SATU_KALI:
        if aturan["tahun"] != tahun_y:
            continue
        cocok = np.ones(n, dtype=bool)
        if aturan.get("kddept") is not None:
            cocok &= (kddept_arr == aturan["kddept"])
        if aturan.get("jenis_belanja") is not None:
            cocok &= (jenis_arr == aturan["jenis_belanja"])
        mask[cocok, aturan["bulan"] - 1] = True
    return mask


def _cocokkan_kebijakan(kddept: int, akun: str, tahun_y: int) -> list:
    """Cari semua aturan kebijakan yang cocok untuk kddept+akun+tahun tertentu."""
    akun_lower = str(akun).lower()
    hasil = []
    for aturan in KONFIGURASI_KEBIJAKAN:
        if aturan["tahun_efektif"] != tahun_y:
            continue
        if aturan["kddept"] is not None and aturan["kddept"] != kddept:
            continue
        if aturan["akun_pola"].lower() not in akun_lower:
            continue
        hasil.append(aturan)
    return hasil


def _terapkan_kebijakan(
    baseline_bulanan: np.ndarray, bulan_penuh_terakhir: int, kddept: int, akun: str, tahun_y: int,
) -> tuple:
    """Terapkan penyesuaian kebijakan ke array baseline forecast 12 bulan (hanya bulan-bulan
    yang MASIH diproyeksikan, yaitu index >= bulan_penuh_terakhir, karena bulan yang sudah
    jadi aktual tidak boleh diubah). Return (array_baru, daftar_keterangan_penyesuaian).

    Logika per aturan: baseline forecast bulan m (m 1-indexed) diasumsikan merefleksikan
    indeks_lama (karena dihitung dari histori sebelum kenaikan berlaku). Untuk bulan
    >= bulan_rapel: dikalikan (indeks_baru/indeks_lama). Untuk bulan_rapel itu sendiri,
    DITAMBAH rapel = akumulasi selisih (indeks_baru-indeks_lama)/indeks_lama x baseline
    bulan-bulan bulan_mulai..bulan_rapel-1 (yang secara substansi sudah harus naik tapi
    belum dibayar). Bulan antara bulan_mulai..bulan_rapel-1 sendiri TETAP di indeks lama
    (baru dibayar penuh + rapel saat bulan_rapel), sesuai praktik umum pencairan rapel."""
    hasil = baseline_bulanan.copy()
    keterangan = []
    aturan_cocok = _cocokkan_kebijakan(kddept, akun, tahun_y)
    for aturan in aturan_cocok:
        idx_lama, idx_baru = aturan["indeks_lama"], aturan["indeks_baru"]
        bln_mulai, bln_rapel = aturan["bulan_mulai"], aturan["bulan_rapel"]
        faktor = idx_baru / idx_lama if idx_lama else 1.0
        rapel_total = 0.0
        for m in range(bln_mulai, bln_rapel):  # bulan2 sebelum rapel diproses (index 1..12)
            if m - 1 >= bulan_penuh_terakhir:  # hanya hitung dari porsi yang masih proyeksi
                rapel_total += hasil[m - 1] * (faktor - 1.0)
        for m in range(bln_rapel, 13):
            if m - 1 < bulan_penuh_terakhir:
                continue
            hasil[m - 1] = hasil[m - 1] * faktor
            if m == bln_rapel:
                hasil[m - 1] += rapel_total
        if rapel_total > 0 or any(m - 1 >= bulan_penuh_terakhir for m in range(bln_mulai, 13)):
            keterangan.append(
                f"{aturan['nama']}: indeks {idx_lama:.0%}→{idx_baru:.0%} efektif bulan "
                f"{bln_mulai}, rapel dibayar bulan {bln_rapel} (+Rp {rapel_total:,.0f})"
            )
    return hasil, keterangan


# --------------------------------------------------------------------------
# 2. Bangun profil historis (tertimbang) untuk sembarang grouping
# --------------------------------------------------------------------------

def _bangun_profil_historis(df_hist: pd.DataFrame, tahun_y: int, group_cols: list) -> pd.DataFrame:
    """Hitung target tahunan tertimbang & profil bulanan tertimbang untuk tiap grup di
    group_cols, dari data histori (df_hist, sudah difilter tahun_y-5 .. tahun_y-1).

    Return DataFrame ber-index group_cols dengan kolom:
      RATE_TERTIMBANG      -- rerata tertimbang (realisasi/pagu) lintas tahun yang tersedia
      RUPIAH_TERTIMBANG    -- rerata tertimbang realisasi rupiah lintas tahun (utk Pegawai)
      PROFIL_JAN..PROFIL_DES -- proporsi bulanan tertimbang, jumlah = 1 (NaN kalau tak ada data)
      CV_RATE               -- koefisien variasi (std/mean) dari rate tahunan antar tahun yang
                                dipakai (dipakai utk confidence score)
      N_TAHUN                -- jumlah tahun histori yang benar-benar dipakai (0-5)
    """
    if df_hist.empty:
        # Tetap kembalikan DataFrame dgn struktur kolom & TIPE DATA yang benar (kosong 0 baris),
        # supaya kode pemanggil (yang mem-merge hasil ini dgn "on=group_cols") tidak KeyError
        # krn kolom kunci merge tidak ada, dan operasi numerik sesudahnya (np.isnan dkk) tidak
        # error krn kolom bertipe object alih2 float. Kasus nyata: tahun_y=2021 (tahun paling
        # awal di data) -> df_hist (tahun_y-5..tahun_y-1) otomatis kosong krn tidak ada data
        # sebelum 2021 sama sekali.
        kolom_numerik = (
            ["RATE_TERTIMBANG", "RUPIAH_TERTIMBANG", "PAGU_TERTIMBANG", "N_TAHUN", "CV_RATE"]
            + [f"PROFIL_{c}" for c in BULAN_KOLOM]
        )
        kosong = pd.DataFrame({c: pd.Series(dtype="float64") for c in kolom_numerik})
        for c in group_cols:
            kosong.insert(0, c, pd.Series(dtype="object"))
        return kosong[group_cols + kolom_numerik]

    g = (
        df_hist.groupby(group_cols + ["TAHUN"], as_index=False)[["PAGU"] + BULAN_KOLOM]
        .sum()
    )
    g["REALISASI"] = g[BULAN_KOLOM].sum(axis=1)
    g["BOBOT"] = (tahun_y - g["TAHUN"]).map(BOBOT_TAHUN)

    # Ambang minimal pagu supaya valid dipakai menghitung RATE (realisasi/pagu). Di data
    # riil kadang muncul pagu berupa angka simbolis/placeholder (mis. Rp 1.000) utk akun yang
    # sebenarnya belum benar2 dianggarkan tahun itu -- kalau tetap dipakai, rate bisa meledak
    # jadi ribuan persen (realisasi jutaan / pagu seribu) dan merusak rerata tertimbang, apalagi
    # utk Belanja Pegawai yang forecast-nya TIDAK dibatasi pagu (lihat hitung_forecast_satker_akun).
    AMBANG_PAGU_MINIMAL = 1_000_000  # Rp 1 juta
    # Rate tahunan juga dibatasi atas (safety net data quality, BUKAN keputusan kebijakan) --
    # realisasi > 300% dari pagu di satu tahun nyaris pasti anomali data, bukan pola wajar yang
    # ingin direplikasi ke tahun proyeksi.
    BATAS_ATAS_RATE_TAHUNAN = 3.0

    rate_valid = g["PAGU"] >= AMBANG_PAGU_MINIMAL
    g["RATE"] = np.where(rate_valid, g["REALISASI"] / g["PAGU"].replace(0, np.nan), np.nan)
    g["RATE"] = g["RATE"].clip(upper=BATAS_ATAS_RATE_TAHUNAN)
    g["W_RATE_NUM"] = np.where(rate_valid, g["BOBOT"] * g["RATE"], 0.0)
    g["W_RATE_DEN"] = np.where(rate_valid, g["BOBOT"], 0.0)
    g["W_RUP_NUM"] = g["BOBOT"] * g["REALISASI"]
    g["W_PAGU_NUM"] = g["BOBOT"] * g["PAGU"]

    prop_valid = g["REALISASI"] > 0
    real_aman = g["REALISASI"].replace(0, np.nan)
    for c in BULAN_KOLOM:
        g[f"WPROP_{c}"] = np.where(prop_valid, g["BOBOT"] * (g[c] / real_aman), 0.0)
    g["WPROP_DEN"] = np.where(prop_valid, g["BOBOT"], 0.0)

    kolom_jumlah = (
        ["W_RATE_NUM", "W_RATE_DEN", "W_RUP_NUM", "W_PAGU_NUM", "BOBOT", "WPROP_DEN"]
        + [f"WPROP_{c}" for c in BULAN_KOLOM]
    )
    agg = g.groupby(group_cols, as_index=False)[kolom_jumlah].sum()
    n_tahun = g.groupby(group_cols)["TAHUN"].nunique().rename("N_TAHUN")
    agg = agg.merge(n_tahun, on=group_cols)

    agg["RATE_TERTIMBANG"] = np.where(
        agg["W_RATE_DEN"] > 0, agg["W_RATE_NUM"] / agg["W_RATE_DEN"], np.nan
    )
    agg["RUPIAH_TERTIMBANG"] = np.where(
        agg["BOBOT"] > 0, agg["W_RUP_NUM"] / agg["BOBOT"], np.nan
    )
    # Rerata pagu tertimbang -- dipakai sbg penjaga skala utk Belanja Pegawai: kalau pagu
    # tahun berjalan jauh lebih kecil drpd rerata historis (indikasi reklasifikasi akun,
    # mis. komponen tunjangan dipindah ke akun lain), target rupiah pegawai perlu ikut
    # diskalakan turun -- lihat pemakaiannya di hitung_forecast_satker_akun.
    agg["PAGU_TERTIMBANG"] = np.where(
        agg["BOBOT"] > 0, agg["W_PAGU_NUM"] / agg["BOBOT"], np.nan
    )
    for c in BULAN_KOLOM:
        agg[f"PROFIL_{c}"] = np.where(
            agg["WPROP_DEN"] > 0, agg[f"WPROP_{c}"] / agg["WPROP_DEN"], np.nan
        )
    profil_cols = [f"PROFIL_{c}" for c in BULAN_KOLOM]
    total_profil = agg[profil_cols].sum(axis=1)
    for c in profil_cols:
        agg[c] = np.where(total_profil > 0, agg[c] / total_profil, np.nan)

    # Koefisien variasi tingkat realisasi tahunan antar tahun yang dipakai -- proxy kestabilan
    # pola historis (dipakai utk confidence score). Weighted std sederhana (populasi).
    rate_pivot = g.pivot_table(index=group_cols, columns="TAHUN", values="RATE")
    bobot_pivot = g.pivot_table(index=group_cols, columns="TAHUN", values="BOBOT")
    mean_w = agg.set_index(group_cols)["RATE_TERTIMBANG"]
    var_num = ((rate_pivot.sub(mean_w, axis=0)) ** 2 * bobot_pivot).sum(axis=1)
    var_den = bobot_pivot.where(rate_pivot.notna()).sum(axis=1)
    var_w = (var_num / var_den).replace([np.inf, -np.inf], np.nan)
    cv = (np.sqrt(var_w) / mean_w.replace(0, np.nan)).clip(lower=0)
    agg = agg.merge(cv.rename("CV_RATE").reset_index(), on=group_cols, how="left")

    keep = (
        group_cols
        + ["RATE_TERTIMBANG", "RUPIAH_TERTIMBANG", "PAGU_TERTIMBANG", "N_TAHUN", "CV_RATE"]
        + profil_cols
    )
    return agg[keep]


def _kelompok_pagu(pagu: pd.Series, dept_jenis: pd.Series) -> pd.Series:
    """Bucket pagu jadi 4 kelompok (Kecil/Sedang/Besar/Sangat Besar) berdasarkan kuartil
    DALAM kelompok (kementerian, jenis belanja) yang sama di tahun ybs -- supaya "sejenis"
    benar-benar dibandingkan dengan skala anggaran yang sepadan, bukan lintas jenis belanja
    yang skalanya jauh berbeda (mis. Belanja Modal vs Belanja Barang)."""
    label = ["Kecil", "Sedang", "Besar", "Sangat Besar"]

    def _qcut_aman(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s, q=4, labels=label, duplicates="drop")
        except (ValueError, IndexError):
            return pd.Series(label[-1], index=s.index)

    return pagu.groupby(dept_jenis).transform(_qcut_aman).astype(str)


# --------------------------------------------------------------------------
# 3. Orkestrasi utama: forecast per kombinasi (satker, akun)
# --------------------------------------------------------------------------

def tentukan_bulan_penuh_terakhir_global(df_all: pd.DataFrame, tahun_y: int) -> tuple:
    """Tentukan bulan terakhir yang datanya sudah LENGKAP (bukan sekadar 'sebelum hari ini').

    Awalnya fungsi ini memakai patokan kalender (bulan berjalan - 1), dengan asumsi data bulan
    berjalan pasti belum lengkap krn bulan itu sendiri belum selesai. Asumsi itu TERBUKTI SALAH
    utk data ini -- realisasi (terutama Belanja Pegawai/gaji) sering sudah tercatat PENUH di
    awal bulan berjalan (gaji dibayar & dicatat di awal bulan utk 1 bulan penuh), jauh sebelum
    tanggal kalender bulan itu berakhir. Kalau tetap dipatok kalender, bulan yg SEBENARNYA sudah
    penuh (mis. Agustus) masih dianggap 'belum penuh' dan dihitung ulang via forecast -- padahal
    aktualnya sudah ada & akurat, sehingga forecast malah menggantikan angka yang sudah benar
    dgn estimasi yang keliru.

    Solusinya: tentukan kelengkapan dari DATA itu sendiri, bukan tanggal. Suatu bulan dianggap
    'penuh' kalau total realisasi bulan itu (seluruh satker&akun) sudah sepadan (>= 50%) dgn
    rata-rata 3 bulan sebelumnya -- bulan yang baru mulai terisi (laporan sedang berjalan,
    trickle sebagian kecil) akan jauh di bawah ambang ini dan tetap dianggap belum penuh."""
    AMBANG_KELENGKAPAN = 0.5
    d = df_all[df_all["TAHUN"] == tahun_y]
    monthly = d[BULAN_KOLOM].sum()
    bulan_terisi = [i + 1 for i, v in enumerate(monthly.values) if v != 0]
    bulan_terakhir = max(bulan_terisi) if bulan_terisi else 0

    bulan_penuh_terakhir = 0
    for m in range(1, 13):
        nilai_bulan = monthly.iloc[m - 1]
        if nilai_bulan == 0:
            break
        bulan_sebelumnya = [b for b in range(max(1, m - 3), m)]
        rata2_sebelumnya = monthly.iloc[[b - 1 for b in bulan_sebelumnya]].mean() if bulan_sebelumnya else nilai_bulan
        if rata2_sebelumnya <= 0 or nilai_bulan >= AMBANG_KELENGKAPAN * rata2_sebelumnya:
            bulan_penuh_terakhir = m
        else:
            break  # bulan ini masih trickle/parsial -> berhenti, bulan setelahnya jg belum penuh
    return bulan_terakhir, bulan_penuh_terakhir


@st.cache_data(show_spinner="Menghitung proyeksi hybrid...")
def hitung_forecast_satker_akun(df_all: pd.DataFrame, tahun_y: int) -> pd.DataFrame:
    """Fungsi inti: hitung forecast 12-bulan hybrid untuk SETIAP kombinasi (KDSATKER, AKUN)
    yang aktif (punya baris data) di tahun_y. Di-cache karena dipanggil ulang tiap halaman
    dibuka/filter berubah, tapi hasilnya sama selama df_all & tahun_y sama.

    Return DataFrame satu baris per (KDSATKER, AKUN) dengan kolom:
      KDDEPT, NMDEPT, KDSATKER, NMSATKER, JENIS BELANJA, LABEL_JENIS_BELANJA, AKUN, PAGU
      AKT_JAN..AKT_DES     -- realisasi aktual tahun_y per bulan (apa adanya dari data)
      HASIL_JAN..HASIL_DES -- aktual (utk bulan <= bulan_penuh_terakhir) atau forecast
      SUMBER_PROFIL         -- 'sendiri' / 'sejenis' / 'rata2_akun' / 'runrate'
      CONFIDENCE            -- skor keyakinan 0-100 (lihat _skor_confidence)
      N_TAHUN_HISTORI        -- jumlah tahun histori yang dipakai (0 kalau full fallback)
      KETERANGAN_KEBIJAKAN   -- teks penyesuaian kebijakan yang diterapkan (kalau ada)
    """
    df_now = df_all[df_all["TAHUN"] == tahun_y].copy()
    if df_now.empty:
        return pd.DataFrame()

    meta_cols = ["KDDEPT", "NMDEPT", "NMSATKER", "JENIS BELANJA", "LABEL_JENIS_BELANJA"]
    now = (
        df_now.groupby(["KDSATKER", "AKUN"], as_index=False)
        .agg({**{c: "first" for c in meta_cols}, "PAGU": "sum", **{c: "sum" for c in BULAN_KOLOM}})
    )
    now = now.rename(columns={c: f"AKT_{c}" for c in BULAN_KOLOM})

    _, bulan_penuh_terakhir = tentukan_bulan_penuh_terakhir_global(df_all, tahun_y)

    # bulan_penuh_terakhir di atas murni data-driven (lihat catatan di
    # tentukan_bulan_penuh_terakhir_global) -- suatu bulan bisa dianggap "penuh" hanya krn
    # komponen besar spt gaji pokok sudah tercatat penuh di awal bulan, PADAHAL bulan kalender
    # itu SENDIRI belum berakhir & komponen lain yang lebih kecil (uang makan, lembur, dst)
    # masih mungkin nambah. Kalau bulan_penuh_terakhir dipakai apa adanya sbg batas
    # aktual-vs-forecast di bawah, bulan kalender yang sedang berjalan akan ditampilkan sbg
    # HASIL = aktual murni (parsial, blm mencakup sisa bulan) -- padahal secara visual (lihat
    # bulan_penuh_terakhir_tampilan di view_dashboard_satker.py) bulan itu tetap ditandai
    # "belum final". bulan_forecast_mulai menyamakan patokan itu: bulan kalender yang sedang
    # berjalan SELALU ikut diproyeksikan (dgn jaring pengaman "tidak boleh lebih kecil dari
    # aktual sejauh ini" di bawah), sehingga isi tabel proyeksi & indikator visual (warna
    # kuning) konsisten -- bulan berjalan selalu menampilkan proyeksi sd akhir bulan, bukan
    # cuma realisasi sejauh ini.
    _hari_ini = date.today()
    if tahun_y == _hari_ini.year:
        bulan_forecast_mulai = min(bulan_penuh_terakhir, _hari_ini.month - 1)
    elif tahun_y > _hari_ini.year:
        bulan_forecast_mulai = 0
    else:
        bulan_forecast_mulai = bulan_penuh_terakhir
    bulan_forecast_mulai = max(bulan_forecast_mulai, 0)

    df_hist = df_all[(df_all["TAHUN"] >= tahun_y - 5) & (df_all["TAHUN"] < tahun_y)]

    # --- Level 1: profil sendiri per (satker, akun) ---
    profil_sendiri = _bangun_profil_historis(df_hist, tahun_y, ["KDSATKER", "AKUN"])

    # --- Level 2: profil "sejenis" per (kementerian, jenis belanja, kelompok pagu) ---
    now["_DEPT_JENIS"] = now["KDDEPT"].astype(str) + "|" + now["JENIS BELANJA"].astype(str)
    now["KELOMPOK_PAGU"] = _kelompok_pagu(now["PAGU"], now["_DEPT_JENIS"])
    # beri label kelompok pagu yang sama ke baris histori, berdasarkan kuartil PAGU tahun
    # berjalan MASING-MASING baris histori dalam kelompok (kementerian,jenis,tahun)-nya sendiri
    # (relatif thd tahunnya sendiri -- pendekatan wajar krn skala pagu naik dari tahun ke tahun)
    df_hist = df_hist.copy()
    df_hist["_DEPT_JENIS"] = df_hist["KDDEPT"].astype(str) + "|" + df_hist["JENIS BELANJA"].astype(str)
    df_hist["_DEPT_JENIS_THN"] = df_hist["_DEPT_JENIS"] + "|" + df_hist["TAHUN"].astype(str)
    df_hist["KELOMPOK_PAGU"] = _kelompok_pagu(df_hist["PAGU"], df_hist["_DEPT_JENIS_THN"])
    profil_cohort = _bangun_profil_historis(
        df_hist, tahun_y, ["KDDEPT", "JENIS BELANJA", "KELOMPOK_PAGU"]
    )

    # --- Level 3: profil rata-rata seluruh satker per AKUN ---
    profil_global_akun = _bangun_profil_historis(df_hist, tahun_y, ["AKUN"])

    # --- Level 4 (jaring pengaman terakhir): rata-rata seluruh satker per JENIS BELANJA ---
    profil_global_jenis = _bangun_profil_historis(df_hist, tahun_y, ["JENIS BELANJA"])

    profil_cols = [f"PROFIL_{c}" for c in BULAN_KOLOM]
    hasil_cols = (
        ["RATE_TERTIMBANG", "RUPIAH_TERTIMBANG", "PAGU_TERTIMBANG", "N_TAHUN", "CV_RATE"] + profil_cols
    )

    gabung = now.merge(profil_sendiri, on=["KDSATKER", "AKUN"], how="left")
    ada_sendiri = gabung["N_TAHUN"].fillna(0) > 0
    gabung["SUMBER_PROFIL"] = np.where(ada_sendiri, "sendiri", "")

    def _isi_dari(fallback_df, on_cols, nama_sumber):
        nonlocal gabung
        butuh = gabung["SUMBER_PROFIL"] == ""
        if not butuh.any() or fallback_df.empty:
            return
        fb = fallback_df.rename(columns={c: f"_FB_{c}" for c in hasil_cols})
        gabung = gabung.merge(fb, on=on_cols, how="left")
        terisi = butuh & gabung["_FB_N_TAHUN"].fillna(0) > 0
        for c in hasil_cols:
            gabung.loc[terisi, c] = gabung.loc[terisi, f"_FB_{c}"]
        gabung.loc[terisi, "SUMBER_PROFIL"] = nama_sumber
        gabung.drop(columns=[f"_FB_{c}" for c in hasil_cols], inplace=True)

    _isi_dari(profil_cohort, ["KDDEPT", "JENIS BELANJA", "KELOMPOK_PAGU"], "sejenis")
    _isi_dari(profil_global_akun, ["AKUN"], "rata2_akun")
    _isi_dari(profil_global_jenis, ["JENIS BELANJA"], "rata2_jenis_belanja")
    gabung["SUMBER_PROFIL"] = gabung["SUMBER_PROFIL"].replace("", "runrate")
    gabung["N_TAHUN"] = gabung["N_TAHUN"].fillna(0)
    gabung["CV_RATE"] = gabung["CV_RATE"].fillna(1.0)  # tanpa histori -> anggap variasi tinggi

    # --- Hitung target tahunan & forecast bulanan (vektor, per baris) ---
    is_pegawai = gabung["JENIS BELANJA"] == 51

    # Target tahunan Belanja Pegawai: SAMA formulanya dgn jenis belanja lain (rate realisasi
    # tertimbang x pagu tahun berjalan), BUKAN rerata rupiah absolut. Ini penting supaya target
    # otomatis ikut menyesuaikan saat pagu berubah signifikan dari tahun ke tahun -- baik naik
    # drastis (mis. penambahan formasi/kenaikan tunjangan besar yg sudah dianggarkan) maupun
    # turun drastis (mis. akun direklasifikasi, realisasinya dipindah ke akun lain). Realisasi
    # gaji PNS/PPPK biasanya sangat stabil thd pagu (~99% terserap tiap tahun), jadi rate
    # tertimbang adalah prediktor yang jauh lebih andal drpd rupiah absolut yg "buta" thd
    # perubahan skala pagu. Pegawai TETAP boleh melebihi pagu (rate historis kadang > 100%,
    # dan lihat juga override target_tersirat di bawah) -- yg beda dari kategori lain hanyalah
    # TIDAK dibatasi maksimal pagu saat proyeksi (lihat langkah "cap" di bawah).
    target_tahunan = gabung["RATE_TERTIMBANG"] * gabung["PAGU"]
    aktual_mat = gabung[[f"AKT_{c}" for c in BULAN_KOLOM]].to_numpy(dtype=float)
    profil_mat = gabung[profil_cols].to_numpy(dtype=float)
    kddept_arr = gabung["KDDEPT"].to_numpy()
    jenis_arr = gabung["JENIS BELANJA"].to_numpy()

    aktual_sd_sekarang = aktual_mat[:, :bulan_forecast_mulai].sum(axis=1) if bulan_forecast_mulai else np.zeros(len(gabung))

    # --- Override target tahunan kalau kecepatan belanja riil tahun berjalan sudah melampaui
    # pola historis (mis. ada kenaikan pagu/kebijakan riil yang belum tertangkap di rerata 5
    # tahun). PENTING: override ini TIDAK memakai rerata rata (aktual/bulan berjalan x12) --
    # itu naif dan gagal memperhitungkan bulan-bulan yang secara historis memang selalu njomplang
    # (THR bulan Maret, gaji ke-13 bulan Juni, dst). Sebagai gantinya, dipakai "target tersirat"
    # = aktual sejauh ini dibagi PROPORSI KUMULATIF HISTORIS pada rentang bulan yang sama --
    # jadi kalau bulan berjalan memang secara historis berat di depan (mis. krn THR), pembagi
    # ikut besar dan tidak salah dikira sebagai kecepatan belanja tahunan yang tinggi.
    #
    # Bulan yang ditandai PENYESUAIAN_SATU_KALI (anomali tidak berulang, mis. rapel gaji
    # keterlambatan) DIKECUALIKAN dari perhitungan target tersirat ini (baik dari aktual
    # pembilang maupun dari proporsi historis penyebut) supaya satu bulan anomali tidak
    # menyeret proyeksi bulan-bulan normal lainnya ikut naik -- tapi uangnya sendiri tetap
    # dihitung penuh saat menentukan SISA target yang harus didistribusikan (lihat di bawah).
    mask_anomali = _masker_anomali_satu_kali(kddept_arr, jenis_arr, tahun_y)
    if bulan_forecast_mulai > 0:
        mask_anomali_sd_sekarang = mask_anomali[:, :bulan_forecast_mulai]
        aktual_bersih_sd_sekarang = np.where(
            mask_anomali_sd_sekarang, 0.0, aktual_mat[:, :bulan_forecast_mulai]
        ).sum(axis=1)
        profil_sd_sekarang = np.nan_to_num(profil_mat[:, :bulan_forecast_mulai], nan=0.0)
        kum_profil_bersih = np.where(mask_anomali_sd_sekarang, 0.0, profil_sd_sekarang).sum(axis=1)
        # Hanya dipakai kalau proporsi historis yg tersisa cukup besar (>=15% dari setahun) --
        # di bawah itu pembagian terlalu sensitif/tidak stabil utk dijadikan dasar ekstrapolasi.
        cukup_data = kum_profil_bersih >= 0.15
        target_tersirat = np.where(
            cukup_data, np.divide(aktual_bersih_sd_sekarang, kum_profil_bersih, out=np.zeros(len(gabung)), where=cukup_data),
            np.nan,
        )
    else:
        target_tersirat = np.full(len(gabung), np.nan)

    # Fallback run-rate sederhana -- HANYA dipakai kalau bahkan target historis (RATE/RUPIAH
    # tertimbang) tidak tersedia sama sekali (NaN, mis. akun benar2 baru tanpa histori apa pun
    # di level manapun) DAN target tersirat di atas juga tidak bisa dihitung.
    rerata_berjalan = np.where(bulan_forecast_mulai > 0, aktual_sd_sekarang / max(bulan_forecast_mulai, 1), 0.0)
    target_runrate_polos = rerata_berjalan * 12
    target_tahunan = np.where(
        np.isnan(target_tahunan),
        np.where(~np.isnan(target_tersirat), target_tersirat, target_runrate_polos),
        target_tahunan,
    )
    # Target historis tidak boleh lebih kecil dari target tersirat riil tahun berjalan --
    # TAPI dinonaktifkan sbg override umum (lihat catatan di bawah). CATATAN PENTING (temuan
    # dari kasus nyata satker 656361/KPU Kota Pekanbaru): utk akun dgn histori pendek/baru
    # (mis. Belanja Pegawai PPPK yang baru ada 2 tahun & jumlah pegawainya masih terus
    # bertambah/ramp-up di tahun2 itu), profil bulanan historisnya "berat di belakang" (porsi
    # bulan awal tahun kecil krn dulu pegawainya masih sedikit). Kalau tahun berjalan ternyata
    # jumlah pegawainya SUDAH STABIL sejak awal tahun (bukan ramp-up lagi), target_tersirat
    # (aktual/proporsi kumulatif historis) salah membaca situasi ini sbg "percepatan belanja"
    # dan mendorong proyeksi sisa tahun naik signifikan -- padahal polanya sebenarnya sudah
    # rata/flat. Override ini terbukti menghasilkan proyeksi yang keliru pada kasus nyata
    # tsb, sehingga utk saat ini TIDAK dipakai sbg override target_tahunan yang valid; target
    # tersirat hanya dipakai sbg fallback TERAKHIR kalau RATE/RUPIAH tertimbang benar2 tidak
    # tersedia sama sekali (lihat blok di atas).
    #
    # PLAFON (bukan lantai) target_tahunan: kalau target berbasis histori (RATE_TERTIMBANG,
    # biasanya dari fallback "sejenis"/cohort krn satker ybs belum/tidak punya histori sendiri)
    # menyiratkan percepatan belanja jauh di atas kecepatan REALISASI RIIL tahun berjalan (mis.
    # cohort peer rata2 menghabiskan ~100% pagu tapi satker ybs baru merealisasikan ~25% dgn pola
    # yang konsisten flat), target itu kemungkinan besar tidak representatif utk satker ybs --
    # dibanding kasus di atas (yang soal target_tersirat dipakai sbg BATAS BAWAH & mendorong
    # target naik saat pola aslinya flat), pemakaian sbg PLAFON di sini hanya menahan target agar
    # TIDAK MELEBIHI apa yang tersirat dari kecepatan riil -- tidak pernah mendorongnya naik --
    # sehingga tidak mengulang kegagalan kasus PPPK tsb (yang gagal krn arah override-nya ke ATAS).
    target_tahunan = np.where(
        np.isnan(target_tersirat), target_tahunan, np.minimum(target_tahunan, target_tersirat)
    )

    # LANTAI khusus Belanja Pegawai: gaji/tunjangan bersifat RUTIN BULANAN, bukan pengeluaran
    # satu-kali. Kadang target berbasis RATE_TERTIMBANG x PAGU sudah TERLAMPAUI oleh realisasi
    # aktual sd sekarang (mis. krn PAGU akun ybs kebetulan lebih kecil dari pola realisasi
    # riilnya tahun ini) -- kalau dibiarkan, sisa_target jadi 0 dan bulan-bulan berikutnya
    # diproyeksikan BERHENTI TOTAL (Rp0), padahal gaji/tunjangan rutin nyaris pasti tetap
    # dibayar. Untuk Pegawai saja, target tidak boleh tersirat lebih kecil dari proyeksi datar
    # (rerata realisasi bulan berjalan x12). Kategori lain (Barang/Modal) SENGAJA tidak diberi
    # lantai ini krn pola belanjanya lazim tidak rata sepanjang tahun (mis. pengadaan besar di
    # awal tahun lalu mereda) -- lihat juga alasan Pegawai tidak dibatasi pagu saat cap di bawah.
    target_tahunan = np.where(
        is_pegawai.to_numpy(), np.maximum(target_tahunan, target_runrate_polos), target_tahunan
    )

    hasil_mat = aktual_mat.copy()
    keterangan_list = [""] * len(gabung)
    if bulan_forecast_mulai < 12:
        sisa_target = np.maximum(target_tahunan - aktual_sd_sekarang, 0.0)
        profil_sisa = profil_mat[:, bulan_forecast_mulai:]
        # normalisasi ulang profil HANYA utk bulan-bulan tersisa (inti dari "rolling forecast")
        profil_sisa = np.nan_to_num(profil_sisa, nan=0.0)
        total_sisa = profil_sisa.sum(axis=1, keepdims=True)
        n_bulan_sisa = 12 - bulan_forecast_mulai
        profil_sisa_norm = np.divide(
            profil_sisa, total_sisa, out=np.full_like(profil_sisa, 1.0 / n_bulan_sisa), where=total_sisa > 0
        )
        forecast_depan = profil_sisa_norm * sisa_target[:, None]

        # Penyesuaian kebijakan -- hanya baris Pegawai yang match aturan (jumlah baris relatif
        # sedikit, loop per baris di sini aman performanya).
        idx_pegawai = np.where(is_pegawai.to_numpy())[0]
        if len(idx_pegawai) and KONFIGURASI_KEBIJAKAN:
            kddept_arr = gabung["KDDEPT"].to_numpy()
            akun_arr = gabung["AKUN"].to_numpy()
            for i in idx_pegawai:
                if not _cocokkan_kebijakan(kddept_arr[i], akun_arr[i], tahun_y):
                    continue
                baseline_12 = hasil_mat[i].copy()
                baseline_12[bulan_forecast_mulai:] = forecast_depan[i]
                baru, ket = _terapkan_kebijakan(baseline_12, bulan_forecast_mulai, kddept_arr[i], akun_arr[i], tahun_y)
                forecast_depan[i] = baru[bulan_forecast_mulai:]
                if ket:
                    keterangan_list[i] = "; ".join(ket)

        # Jaring pengaman DULU (sebelum cap): kalau bulan yang berstatus "forecast" ternyata
        # sudah punya sebagian/seluruh data aktual (mis. laporan sebagian sudah masuk, atau
        # bulan kalender berjalan yang datanya sudah terisi tapi belum tentu lengkap sebulan
        # penuh -- lihat bulan_forecast_mulai di atas), angka aktual itu jadi batas BAWAH --
        # tidak boleh diperkecil oleh forecast. PENTING: ini dilakukan SEBELUM cap pagu (bukan
        # sesudah, spt versi sebelumnya) supaya cap berikutnya bisa menghitung ulang dgn benar &
        # total akhir tetap terjamin <= pagu utk kategori selain Pegawai.
        aktual_sisa = aktual_mat[:, bulan_forecast_mulai:]
        forecast_depan = np.maximum(forecast_depan, aktual_sisa)

        # Cap maksimal pagu utk kategori selain Belanja Pegawai. Bagian yang SUDAH dikunci oleh
        # data aktual (aktual_sisa) tidak boleh dikurangi (itu fakta, bukan estimasi) -- yang
        # diskalakan turun HANYA porsi estimasi murni (forecast_depan - aktual_sisa) kalau
        # totalnya (aktual sd sekarang + aktual_sisa + estimasi murni) akan melebihi pagu.
        pagu_arr = gabung["PAGU"].to_numpy(dtype=float)
        sisa_pagu = np.maximum(pagu_arr - aktual_sd_sekarang, 0.0)
        bagian_estimasi_murni = forecast_depan - aktual_sisa  # >= 0, sudah di-floor di atas
        sisa_pagu_utk_estimasi = np.maximum(sisa_pagu - aktual_sisa.sum(axis=1), 0.0)
        total_estimasi_murni = bagian_estimasi_murni.sum(axis=1)
        perlu_skala = (~is_pegawai.to_numpy()) & (total_estimasi_murni > sisa_pagu_utk_estimasi) & (total_estimasi_murni > 0)
        faktor_skala = np.ones(len(gabung))
        faktor_skala[perlu_skala] = np.clip(
            sisa_pagu_utk_estimasi[perlu_skala] / total_estimasi_murni[perlu_skala], 0.0, 1.0
        )
        forecast_depan = aktual_sisa + bagian_estimasi_murni * faktor_skala[:, None]

        hasil_mat[:, bulan_forecast_mulai:] = forecast_depan

    for i, c in enumerate(BULAN_KOLOM):
        gabung[f"HASIL_{c}"] = hasil_mat[:, i]

    gabung["CONFIDENCE"] = _skor_confidence(gabung["CV_RATE"], gabung["N_TAHUN"], gabung["SUMBER_PROFIL"])
    gabung["KETERANGAN_KEBIJAKAN"] = keterangan_list

    # Keterangan anomali satu-kali yg berlaku utk baris ini (transparansi -- bulan mana yg
    # dikecualikan dari perhitungan kecepatan belanja saat menentukan target tahunan).
    keterangan_anomali = [""] * len(gabung)
    if mask_anomali.any():
        for aturan in PENYESUAIAN_SATU_KALI:
            if aturan["tahun"] != tahun_y:
                continue
            cocok = np.ones(len(gabung), dtype=bool)
            if aturan.get("kddept") is not None:
                cocok &= (kddept_arr == aturan["kddept"])
            if aturan.get("jenis_belanja") is not None:
                cocok &= (jenis_arr == aturan["jenis_belanja"])
            for i in np.where(cocok)[0]:
                keterangan_anomali[i] = aturan["nama"]
    gabung["KETERANGAN_ANOMALI"] = keterangan_anomali
    gabung["BULAN_PENUH_TERAKHIR"] = bulan_penuh_terakhir

    kolom_akhir = (
        ["KDDEPT", "NMDEPT", "KDSATKER", "NMSATKER", "JENIS BELANJA", "LABEL_JENIS_BELANJA",
         "AKUN", "PAGU", "SUMBER_PROFIL", "CONFIDENCE", "N_TAHUN", "KETERANGAN_KEBIJAKAN",
         "KETERANGAN_ANOMALI", "BULAN_PENUH_TERAKHIR"]
        + [f"AKT_{c}" for c in BULAN_KOLOM]
        + [f"HASIL_{c}" for c in BULAN_KOLOM]
    )
    return gabung[kolom_akhir]


def _skor_confidence(cv: pd.Series, n_tahun: pd.Series, sumber: pd.Series) -> pd.Series:
    """Skor keyakinan 0-100 dari koefisien variasi pola historis: makin stabil (CV kecil)
    makin tinggi skornya. Dikurangi (penalti) kalau histori yang dipakai tipis (<3 tahun)
    atau berasal dari fallback (bukan profil satker itu sendiri), karena keduanya menambah
    ketidakpastian di luar yang tertangkap oleh CV semata."""
    skor_dasar = 100.0 / (1.0 + cv.clip(lower=0))
    penalti_tahun = np.where(n_tahun >= 3, 0, np.where(n_tahun > 0, 15, 30))
    penalti_sumber = sumber.map({
        "sendiri": 0, "sejenis": 10, "rata2_akun": 20, "rata2_jenis_belanja": 25, "runrate": 35,
    }).fillna(35)
    skor = (skor_dasar - penalti_tahun - penalti_sumber).clip(lower=5, upper=99)
    return skor.round(1)


# --------------------------------------------------------------------------
# 4. Agregasi hasil ke level tampilan dashboard (per entitas/kategori terpilih)
# --------------------------------------------------------------------------

def agregasi_per_kategori(df_forecast: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    """Filter df_forecast (hasil hitung_forecast_satker_akun) sesuai entitas terpilih
    (KDDEPT dan/atau KDSATKER), lalu jumlahkan per LABEL_JENIS_BELANJA. Return DataFrame
    berindeks LABEL_JENIS_BELANJA dengan kolom bulanan (JAN..DES, angka campuran
    aktual+forecast) + PAGU + CONFIDENCE (rata-rata tertimbang pagu)."""
    d = df_forecast
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            d = d[d[kolom] == nilai]
    if d.empty:
        return pd.DataFrame()

    hasil_cols = [f"HASIL_{c}" for c in BULAN_KOLOM]
    agg = d.groupby("LABEL_JENIS_BELANJA")[["PAGU"] + hasil_cols].sum()
    agg.columns = ["PAGU"] + BULAN_KOLOM

    # confidence per kategori = rata-rata tertimbang pagu antar kombinasi satker-akun
    bobot = d["PAGU"].clip(lower=1)
    conf = (
        (d["CONFIDENCE"] * bobot).groupby(d["LABEL_JENIS_BELANJA"]).sum()
        / bobot.groupby(d["LABEL_JENIS_BELANJA"]).sum()
    )
    agg["CONFIDENCE"] = conf
    return agg


def agregasi_aktual_per_kategori(df_forecast: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    """Sama seperti agregasi_per_kategori tapi memakai kolom AKT_ (murni realisasi aktual,
    tanpa campuran forecast) -- dipakai utk baris 'Total Realisasi' di tabel."""
    d = df_forecast
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            d = d[d[kolom] == nilai]
    if d.empty:
        return pd.DataFrame()
    akt_cols = [f"AKT_{c}" for c in BULAN_KOLOM]
    agg = d.groupby("LABEL_JENIS_BELANJA")[akt_cols].sum()
    agg.columns = BULAN_KOLOM
    return agg


# --------------------------------------------------------------------------
# 5. Early Warning System
# --------------------------------------------------------------------------

def deteksi_early_warning(
    df_forecast: pd.DataFrame, filter_dict: dict, df_all: pd.DataFrame, tahun_y: int,
) -> dict:
    """Deteksi 3 jenis sinyal risiko pada entitas terpilih (agregat semua kategori):
      1. Forecast melebihi pagu (kategori selain Belanja Pegawai)
      2. Penyerapan terlalu cepat/lambat dibanding rata-rata historis pada bulan yang sama
      3. Lonjakan realisasi bulan terakhir yang tidak wajar (di luar pola historis)
    Return dict berisi daftar peringatan (masing2 dgn alasan) + skor risiko agregat 0-100
    dan kategorinya (Rendah/Sedang/Tinggi)."""
    d = df_forecast
    for kolom, nilai in filter_dict.items():
        if nilai is not None:
            d = d[d[kolom] == nilai]
    if d.empty:
        return {"peringatan": [], "skor_risiko": 0, "kategori_risiko": "Rendah"}

    bulan_penuh_terakhir = int(d["BULAN_PENUH_TERAKHIR"].iloc[0])
    peringatan = []
    poin_risiko = 0

    # --- 1. Forecast melebihi pagu (per kategori, kecuali Belanja Pegawai) ---
    hasil_cols = [f"HASIL_{c}" for c in BULAN_KOLOM]
    per_kat = d.groupby("LABEL_JENIS_BELANJA").agg(
        PAGU=("PAGU", "sum"), TOTAL_HASIL=(hasil_cols[0], "sum"),
    )
    per_kat["TOTAL_HASIL"] = d.groupby("LABEL_JENIS_BELANJA")[hasil_cols].sum().sum(axis=1)
    for label, row in per_kat.iterrows():
        if label == LABEL_BELANJA_PEGAWAI:
            continue
        if row["PAGU"] > 0 and row["TOTAL_HASIL"] > row["PAGU"] * 1.001:
            selisih = row["TOTAL_HASIL"] - row["PAGU"]
            peringatan.append({
                "jenis": "Proyeksi Melebihi Pagu",
                "tingkat": "Tinggi",
                "kategori": label,
                "alasan": (
                    f"Total realisasi+proyeksi {label} (Rp {row['TOTAL_HASIL']:,.0f}) diperkirakan "
                    f"melebihi pagu (Rp {row['PAGU']:,.0f}) sebesar Rp {selisih:,.0f}."
                ),
            })
            poin_risiko += 25

    # --- 2. Penyerapan terlalu cepat/lambat dibanding pola historis di bulan yang sama ---
    if bulan_penuh_terakhir > 0:
        akt_cols = [f"AKT_{c}" for c in BULAN_KOLOM[:bulan_penuh_terakhir]]
        pagu_total = d["PAGU"].sum()
        realisasi_sekarang = d[akt_cols].sum().sum()
        persen_sekarang = (realisasi_sekarang / pagu_total * 100) if pagu_total else 0

        # ekspektasi historis: proporsi kumulatif profil bulanan tertimbang (dari sumber yang
        # sama yg dipakai per baris) x pagu -- didekati dgn agregasi HASIL yg totalnya = target
        # tahunan, dibandingkan dgn PROFIL kumulatif rata2 tertimbang seluruh baris.
        d_hist = df_all[(df_all["TAHUN"] >= tahun_y - 5) & (df_all["TAHUN"] < tahun_y)]
        d_hist_scope = d_hist
        for kolom, nilai in filter_dict.items():
            if nilai is not None and kolom in d_hist_scope.columns:
                d_hist_scope = d_hist_scope[d_hist_scope[kolom] == nilai]
        if not d_hist_scope.empty:
            g = d_hist_scope.groupby("TAHUN")[["PAGU"] + BULAN_KOLOM].sum()
            g["REALISASI_SD_BULAN"] = g[BULAN_KOLOM[:bulan_penuh_terakhir]].sum(axis=1)
            g["PERSEN_SD_BULAN"] = np.where(g["PAGU"] > 0, g["REALISASI_SD_BULAN"] / g["PAGU"] * 100, np.nan)
            bobot = pd.Series({thn: BOBOT_TAHUN.get(tahun_y - thn, 0) for thn in g.index})
            valid = g["PERSEN_SD_BULAN"].notna()
            if valid.any() and bobot[valid].sum() > 0:
                persen_historis = (g.loc[valid, "PERSEN_SD_BULAN"] * bobot[valid]).sum() / bobot[valid].sum()
                gap = persen_sekarang - persen_historis
                if gap < -15:
                    peringatan.append({
                        "jenis": "Penyerapan Lebih Lambat dari Historis",
                        "tingkat": "Sedang",
                        "kategori": "Total",
                        "alasan": (
                            f"Penyerapan sd bulan ke-{bulan_penuh_terakhir} baru {persen_sekarang:.1f}% "
                            f"dari pagu, sedangkan rata-rata historis pada bulan yang sama "
                            f"{persen_historis:.1f}% (selisih {gap:.1f} poin persen)."
                        ),
                    })
                    poin_risiko += 15
                elif gap > 15:
                    peringatan.append({
                        "jenis": "Penyerapan Lebih Cepat dari Historis",
                        "tingkat": "Sedang",
                        "kategori": "Total",
                        "alasan": (
                            f"Penyerapan sd bulan ke-{bulan_penuh_terakhir} sudah {persen_sekarang:.1f}% "
                            f"dari pagu, jauh di atas rata-rata historis {persen_historis:.1f}% "
                            f"(selisih +{gap:.1f} poin persen) -- perlu dicek apakah wajar."
                        ),
                    })
                    poin_risiko += 10

        # --- 3. Lonjakan tidak wajar di bulan terakhir ---
        # Dilewati kalau bulan ini sudah ditandai sebagai anomali satu-kali yang diketahui
        # sebabnya (lihat PENYESUAIAN_SATU_KALI) -- supaya event yang sudah dijelaskan (mis.
        # rapel keterlambatan gaji) tidak dilaporkan ulang seolah-olah anomali misterius.
        kddept_di_scope = set(d["KDDEPT"].unique())
        anomali_bulan_ini = any(
            a["tahun"] == tahun_y and a["bulan"] == bulan_penuh_terakhir
            and (a.get("kddept") is None or a["kddept"] in kddept_di_scope)
            for a in PENYESUAIAN_SATU_KALI
        )
        bulan_kolom_terakhir = BULAN_KOLOM[bulan_penuh_terakhir - 1]
        nilai_bulan_ini = d[f"AKT_{bulan_kolom_terakhir}"].sum()
        hist_bulan_ini = (
            d_hist_scope.groupby("TAHUN")[bulan_kolom_terakhir].sum()
            if not d_hist_scope.empty else pd.Series(dtype=float)
        )
        if len(hist_bulan_ini) >= 2 and not anomali_bulan_ini:
            mean_h, std_h = hist_bulan_ini.mean(), hist_bulan_ini.std()
            if std_h and std_h > 0 and nilai_bulan_ini > mean_h + 2 * std_h:
                peringatan.append({
                    "jenis": "Lonjakan Realisasi Tidak Wajar",
                    "tingkat": "Sedang",
                    "kategori": "Total",
                    "alasan": (
                        f"Realisasi bulan {bulan_kolom_terakhir} (Rp {nilai_bulan_ini:,.0f}) jauh di "
                        f"atas rata-rata historis bulan yang sama (Rp {mean_h:,.0f} ± {std_h:,.0f})."
                    ),
                })
                poin_risiko += 15

    skor_risiko = min(poin_risiko, 100)
    kategori_risiko = "Tinggi" if skor_risiko >= 50 else ("Sedang" if skor_risiko >= 20 else "Rendah")
    return {"peringatan": peringatan, "skor_risiko": skor_risiko, "kategori_risiko": kategori_risiko}


# --------------------------------------------------------------------------
# 6. Heatmap deviasi antar satker
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def hitung_heatmap_deviasi(
    df_forecast: pd.DataFrame, df_all: pd.DataFrame, tahun_y: int, kddept: int = None, maks_satker: int = 25,
) -> pd.DataFrame:
    """Hitung deviasi (poin persen) antara persen realisasi AKTUAL kumulatif tiap satker
    dengan persen realisasi historis rata-rata tertimbang pada bulan yang sama, per bulan
    sd bulan_penuh_terakhir. Dibatasi ke satker dgn pagu terbesar (maks_satker) supaya
    heatmap tetap terbaca kalau cakupannya luas (mis. seluruh kementerian/seluruh satker)."""
    d = df_forecast if kddept is None else df_forecast[df_forecast["KDDEPT"] == kddept]
    if d.empty:
        return pd.DataFrame()
    bulan_penuh_terakhir = int(d["BULAN_PENUH_TERAKHIR"].iloc[0])
    if bulan_penuh_terakhir == 0:
        return pd.DataFrame()

    satker_pagu = d.groupby(["KDSATKER", "NMSATKER"])["PAGU"].sum().sort_values(ascending=False)
    top_satker = satker_pagu.head(maks_satker).index

    akt_cols = [f"AKT_{c}" for c in BULAN_KOLOM[:bulan_penuh_terakhir]]
    per_satker = d.groupby(["KDSATKER", "NMSATKER"]).agg(PAGU=("PAGU", "sum"))
    akt_kumulatif = d.groupby(["KDSATKER", "NMSATKER"])[akt_cols].sum().cumsum(axis=1)
    persen_aktual = akt_kumulatif.div(per_satker["PAGU"].replace(0, np.nan), axis=0) * 100

    d_hist = df_all[(df_all["TAHUN"] >= tahun_y - 5) & (df_all["TAHUN"] < tahun_y)]
    if kddept is not None:
        d_hist = d_hist[d_hist["KDDEPT"] == kddept]
    satker_ids = [s[0] for s in top_satker]
    d_hist = d_hist[d_hist["KDSATKER"].isin(satker_ids)]

    baris_heatmap = []
    for (kdsatker, nmsatker) in top_satker:
        d_hist_s = d_hist[d_hist["KDSATKER"] == kdsatker]
        if d_hist_s.empty or kdsatker not in persen_aktual.index.get_level_values(0):
            continue
        g = d_hist_s.groupby("TAHUN")[["PAGU"] + BULAN_KOLOM[:bulan_penuh_terakhir]].sum()
        g["KUM"] = g[BULAN_KOLOM[:bulan_penuh_terakhir]].cumsum(axis=1).iloc[:, -1] if bulan_penuh_terakhir else 0
        kumulatif_bulanan = g[BULAN_KOLOM[:bulan_penuh_terakhir]].cumsum(axis=1)
        persen_hist_bulanan = kumulatif_bulanan.div(g["PAGU"].replace(0, np.nan), axis=0) * 100
        bobot = pd.Series({thn: BOBOT_TAHUN.get(tahun_y - thn, 0) for thn in g.index})
        deviasi_baris = []
        for i, bulan in enumerate(BULAN_KOLOM[:bulan_penuh_terakhir]):
            valid = persen_hist_bulanan[bulan].notna()
            if valid.any() and bobot[valid].sum() > 0:
                persen_hist = (persen_hist_bulanan.loc[valid, bulan] * bobot[valid]).sum() / bobot[valid].sum()
            else:
                persen_hist = np.nan
            persen_akt = persen_aktual.loc[(kdsatker, nmsatker), f"AKT_{bulan}"]
            deviasi_baris.append(persen_akt - persen_hist if pd.notna(persen_hist) else np.nan)
        baris_heatmap.append([f"{kdsatker:06d} - {nmsatker}"] + deviasi_baris)

    if not baris_heatmap:
        return pd.DataFrame()
    kolom = ["Satker"] + BULAN_KOLOM[:bulan_penuh_terakhir]
    return pd.DataFrame(baris_heatmap, columns=kolom).set_index("Satker")
