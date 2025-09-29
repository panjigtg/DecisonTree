import streamlit as st
import numpy as np
import joblib
from sklearn import tree
import matplotlib.pyplot as plt

# -------------------------
# Konfigurasi Halaman Streamlit
# -------------------------
# Tema light/dark mode sudah didukung secara bawaan oleh Streamlit.
# Pengguna dapat mengubahnya melalui menu Settings (☰) di pojok kanan atas.
st.set_page_config(
    page_title="Prediksi Resiko Kelulusan Mahasiswa",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Load Model Decision Tree (menggunakan cache agar lebih cepat)
# -------------------------
@st.cache_resource
def load_model():
    # Pastikan path file ini benar sesuai dengan lokasi file Anda
    return joblib.load("./dataset/decision_tree_model_resiko.pkl")

model = load_model()

# -------------------------
# Judul dan Deskripsi Aplikasi
# -------------------------
st.title("🎓 Prediksi Resiko Kelulusan Mahasiswa")
st.markdown("Gunakan dasbor ini untuk memprediksi **resiko kelulusan** mahasiswa berdasarkan performa akademik mereka.")

# -------------------------
# Layout Utama: Kiri untuk Input, Kanan untuk Output
# -------------------------
main_col1, main_col2 = st.columns([0.6, 0.4], gap="large")

# -------------------------
# PANEL INPUT (Kiri)
# -------------------------
with main_col1:
    with st.container(border=True):
        st.header("📝 Masukkan Data Akademik")
        
        with st.form("tree_form"):
            # Membagi form menjadi dua kolom agar tidak terlalu panjang
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                ipk = st.number_input("IPK Terakhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
                mk_tidak_lulus = st.number_input("Jumlah Mata Kuliah Gagal", min_value=0, max_value=20, value=0)
                cuti = st.number_input("Jumlah Semester Cuti", min_value=0, max_value=5, value=0)
                kerja = st.selectbox("Status Bekerja", ["Tidak", "Ya"], help="Apakah mahasiswa bekerja sambil kuliah?")
            
            with form_col2:
                semester = st.number_input("Semester Saat Ini", min_value=1, max_value=20, value=8)
                ips_rata = st.number_input("IPS Rata-rata", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
                ips_akhir = st.number_input("IPS Semester Terakhir", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
                hadir = st.selectbox("Tingkat Kehadiran", ["Tinggi", "Sedang", "Rendah"])

            submit = st.form_submit_button("🔍 Analisis Sekarang", use_container_width=True)

# -------------------------
# PANEL OUTPUT (Kanan)
# -------------------------
with main_col2:
    with st.container(border=True):
        st.header("📊 Hasil Analisis")

        if not submit:
            st.info("Hasil prediksi akan muncul di sini setelah Anda menekan tombol 'Analisis Sekarang'.")
        
        if submit:
            # --- Proses Prediksi (Logika tidak diubah) ---
            kerja_binary = 1 if kerja == "Ya" else 0
            hadir_binary = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}[hadir]
            ips_tren = ips_akhir - ips_rata
            lancar = 1 if semester <= 8 else 0

            input_data = np.array([[ipk, mk_tidak_lulus, cuti, kerja_binary,
                                    semester, ips_rata, ips_akhir,
                                    ips_tren, hadir_binary, lancar]])

            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]
            
            # --- Tampilan Hasil Prediksi ---
            emoji = {"Rendah": "🟢", "Sedang": "🟡", "Tinggi": "🔴"}
            help_text = {
                "Rendah": "Mahasiswa memiliki probabilitas tinggi untuk lulus tepat waktu.",
                "Sedang": "Performa mahasiswa perlu dipantau lebih lanjut.",
                "Tinggi": "Mahasiswa beresiko tinggi untuk tidak lulus tepat waktu."
            }

            st.metric(
                label="Prediksi Resiko Kelulusan",
                value=f"{emoji[pred]} {pred}",
                help=help_text[pred]
            )
            
            st.divider()

            st.markdown("#### Detail Probabilitas")
            
            # Menampilkan probabilitas dengan progress bar
            for i, class_name in enumerate(model.classes_):
                st.markdown(f"**{class_name}**")
                st.progress(proba[i], text=f"{proba[i]*100:.2f}%")
                
# -------------------------
# Penjelasan Resiko dan Visualisasi Pohon (di bawah layout utama)
# -------------------------
st.divider()

# Menggunakan kolom agar penjelasan dan visualisasi bisa berdampingan jika layar lebar
info_col1, info_col2 = st.columns([0.4, 0.6])

with info_col1:
    st.markdown("""
    ### ℹ️ Apa itu *Resiko Kelulusan*?

    Model ini memprediksi kemungkinan mahasiswa **tidak lulus tepat waktu** berdasarkan performa akademiknya.
    - 🟢 **Rendah**: Kemungkinan besar akan lulus tepat waktu.
    - 🟡 **Sedang**: Perlu perhatian dan pemantauan lebih lanjut.
    - 🔴 **Tinggi**: Kemungkinan besar akan terlambat atau gagal lulus.
    """)

with info_col2:
    with st.expander("🌳 Lihat Visualisasi Pohon Keputusan (Decision Tree)"):
        fig = plt.figure(figsize=(20, 10)) # Ukuran disesuaikan agar lebih jelas
        tree.plot_tree(
            model,
            filled=True,
            feature_names=[
                "IPK", "MK Tidak Lulus", "Cuti", "Kerja", "Semester",
                "IPS Rata", "IPS Akhir", "IPS Tren", "Kehadiran", "Lancar"
            ],
            class_names=model.classes_,
            rounded=True,
            fontsize=8 # Ukuran font disesuaikan
        )
        st.pyplot(fig)