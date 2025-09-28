import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset_kelulusan_mahasiswa.csv")

# Label encoding
le = LabelEncoder()
df['Pekerjaan Sambil Kuliah'] = le.fit_transform(df['Pekerjaan Sambil Kuliah'])
df['Kategori Kehadiran'] = le.fit_transform(df['Kategori Kehadiran'])

X = df.drop(columns=['Status Kelulusan'])
y = df['Status Kelulusan']

# Training model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

st.title("Prediksi Status Kelulusan Mahasiswa 🎓")

# Form input
ipk = st.slider("IPK", 0.0, 4.0, 3.0)
mtk_tidak_lulus = st.number_input("Jumlah Mata Kuliah Tidak Lulus", 0)
jumlah_cuti = st.number_input("Jumlah Cuti Akademik", 0)
kerja = st.selectbox("Pekerjaan Sambil Kuliah", ['Ya', 'Tidak'])
jumlah_semester = st.number_input("Jumlah Semester", 1)
ips_rata2 = st.slider("IPS Rata-rata", 0.0, 4.0, 3.0)
ips_sem_akhir = st.slider("IPS Semester Akhir", 0.0, 4.0, 3.0)
ips_tren = st.slider("IPS Tren", -2.0, 2.0, 0.0)
kehadiran = st.selectbox("Kategori Kehadiran", ['Rendah', 'Sedang', 'Tinggi'])

# Convert input to array
kerja_encoded = 1 if kerja == 'Ya' else 0
kehadiran_encoded = le.transform([kehadiran])[0]

input_data = np.array([[
    ipk, mtk_tidak_lulus, jumlah_cuti, kerja_encoded,
    jumlah_semester, ips_rata2, ips_sem_akhir,
    ips_tren, kehadiran_encoded
]])

if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    hasil = "Lulus 🎓" if pred == 1 else "Tidak Lulus ❌"
    st.success(f"Hasil Prediksi: **{hasil}**")
