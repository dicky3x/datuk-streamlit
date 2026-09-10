"""
app.py -- Router utama DATUK
-------------------------------
Menentukan halaman mana yang muncul di navigasi berdasarkan status login:
- Dashboard Pagu dan Realisasi: selalu ada. Kalau login pakai kode satker (bukan super
  user), datanya otomatis dibatasi ke satker itu sendiri (lihat view_dashboard_satker.py).
- Dashboard Prioritas Presiden & Program Strategis: hanya muncul di navigasi kalau satker
  yang login memang punya anggaran terkait (atau kalau login sebagai super user).
- Dashboard Dana Transfer ke Daerah: HANYA muncul & bisa diakses oleh super user.

Login & loading data yang dipakai bersama semua halaman ada di common.py.

Catatan ttg navigasi sebelum/sesudah login:
st.navigation() DIPANGGIL DI SETIAP KONDISI DI BAWAH (baik sebelum maupun sesudah login),
bukan cuma sesudah login. Ini penting karena dua alasan:
1. Begitu SATU SESI SAJA memanggil st.navigation(), Streamlit langsung & PERMANEN (utk
   semua sesi lain, sampai server di-restart) berhenti menampilkan navigasi otomatis bawaan
   (daftar nama file mentah di folder pages/). Kalau st.navigation() cuma dipanggil SETELAH
   login, ada jendela waktu (dari server baru nyala sampai orang pertama berhasil login)
   dimana pengunjung yang BELUM login bisa melihat navigasi otomatis bawaan itu.
2. Dengan memanggil st.navigation() juga sebelum login -- pakai position="hidden" -- navigasi
   tidak digambar sama sekali selama di layar login. Begitu login berhasil, kita panggil
   st.navigation() lagi dengan position="sidebar" (default) sehingga navigasi muncul kembali.
"""

import streamlit as st

from common import get_data, inject_visual_theme, require_login, satker_ada_di_path, klasifikasi_kewenangan

st.set_page_config(page_title="DATUK", page_icon="📊", layout="wide")

# Tema visual global (font Roboto + efek latar particle constellation) -- dipanggil
# SEKALI di sini (entrypoint) supaya berlaku di semua halaman, termasuk layar login.
inject_visual_theme()

df = get_data()

if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    # Belum login -- panggil st.navigation() dengan position="hidden" (bukan di-skip total)
    # supaya navigasi otomatis bawaan Streamlit tidak akan pernah sempat muncul (lihat
    # catatan di atas), sekaligus navigasi kita sendiri pun tidak tampak sampai login sukses.
    halaman_login = st.Page(
        lambda: require_login(df, judul_halaman="DATUK"), title="Login", default=True
    )
    pg = st.navigation([halaman_login], position="hidden")
    pg.run()
    st.stop()

# Login ditangani SEKALI di sini (router). Halaman lain tinggal baca st.session_state.auth,
# tidak perlu panggil require_login() lagi (biar tidak dobel render status login/logout).
auth = require_login(df, judul_halaman="DATUK")
is_super = auth["role"] == "super"
scope_kdsatker = None if is_super else auth["kdsatker"]

pages = [
    st.Page("view_dashboard_satker.py", title="Dashboard Pagu dan Realisasi", icon="📊", default=True),
]

if is_super or satker_ada_di_path("data/prioritas_presiden.csv.gz", scope_kdsatker):
    pages.append(
        st.Page("pages/2_Dashboard_Prioritas_Presiden.py", title="Dashboard Prioritas Presiden", icon="⭐")
    )

if is_super or satker_ada_di_path("data/program_strategis.csv.gz", scope_kdsatker):
    pages.append(
        st.Page("pages/3_Dashboard_Program_Strategis.py", title="Dashboard Program Strategis", icon="🎯")
    )

if is_super:
    pages.append(
        st.Page("pages/4_Dashboard_Dana_Transfer_ke_Daerah.py", title="Dashboard Dana Transfer ke Daerah", icon="🏘️")
    )

# Halaman 5 (Satker Dekon & TP) hanya dimasukkan ke navigasi kalau kolom KEWENANGAN
# memang berisi data yang bisa dipetakan ke Dekonsentrasi/Tugas Pembantuan -- lihat
# catatan di pages/5_Dashboard_Satker_Dekon_TP.py soal kolom ini perlu di-rebuild dulu
# dari file sumber yang memuat info kewenangan.
if "KEWENANGAN" in df.columns and df["KEWENANGAN"].apply(klasifikasi_kewenangan).notna().any():
    pages.append(
        st.Page("pages/5_Dashboard_Satker_Dekon_TP.py", title="Dashboard Satker Dekon & TP", icon="🏢")
    )

pg = st.navigation(pages)  # position="sidebar" (default) -- navigasi tampil
pg.run()
